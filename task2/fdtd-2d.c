#define MEDIUM_DATASET

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <signal.h>
#include <mpi.h>
#include <mpi-ext.h>

#include "fdtd-2d.h"

#define HELPERS 2

int rest_helpers = HELPERS;

double bench_t_start, bench_t_end;

int rank;
int ranksize;
MPI_Request req[4];
MPI_Comm comm_world;

float *ex;
float *ey;
float *hz;
float _fict_[TMAX];
int nrow;

int t;

int killed = 0;

static
double rtclock()
{
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, NULL);
    if (stat != 0)
      printf ("Error return from gettimeofday: %d", stat);
    return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void bench_timer_start()
{
  bench_t_start = rtclock ();
}

void bench_timer_stop()
{
  bench_t_end = rtclock ();
}

void bench_timer_print()
{
  printf ("Time in seconds = %0.6lf\n", bench_t_end - bench_t_start);
}

void dump_iter() {
    char str[30];
    sprintf(str, "dump_%d_time", rank);
    int fd = open(str, O_CREAT | O_WRONLY, 0666);
    write(fd, &t, sizeof(t));
    close(fd);
}

void dump() {
    char str[30];
    sprintf(str, "dump_%d", rank);
    int fd = open(str, O_CREAT | O_WRONLY, 0666);
    write(fd, ex, nrow * NY * sizeof(*ex));
    write(fd, ey, nrow * NY * sizeof(*ey));
    write(fd, hz, nrow * NY * sizeof(*hz));
    close(fd);
}

void load() {
    char str[30];
    sprintf(str, "dump_%d", rank);
    int fd = open(str, O_RDONLY);
    read(fd, ex, nrow * NY * sizeof(*ex));
    read(fd, ey, nrow * NY * sizeof(*ey));
    read(fd, hz, nrow * NY * sizeof(*hz));
    close(fd);
}

void load_iter() {
    char str[30];
    sprintf(str, "dump_%d_time", rank);
    int fd = open(str, O_RDONLY);
    read(fd, &t, sizeof(t));
    close(fd);
}

static
void print_array(int nx,
   int ny,
   float *ex,
   float *ey,
   float *hz)
{
  int i, j;
  double checksum = 0;
  //fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  //fprintf(stderr, "begin dump: %s", "ex");
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++) {
      //if ((i * nx + j) % 20 == 0) fprintf(stdout, "\n");
      //fprintf(stdout, "%0.2f ", *(ex + i * ny + j));
      checksum += *(ex + i * ny + j) / (nx * ny);
    }
  //fprintf(stderr, "\nend   dump: %s\n", "ex");
  //fprintf(stderr, "==END   DUMP_ARRAYS==\n");
  //fprintf(stderr, "begin dump: %s", "ey");
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++) {
      //if ((i * nx + j) % 20 == 0) fprintf(stdout, "\n");
      //fprintf(stdout, "%0.2f ", *(ey + i * ny + j));
      checksum += *(ey + i * ny + j) / (nx * ny);
    }
  //fprintf(stderr, "\nend   dump: %s\n", "ey");

  //fprintf(stderr, "begin dump: %s", "hz");
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++) {
      //if ((i * nx + j) % 20 == 0) fprintf(stderr, "\n");
      //fprintf(stderr, "%0.2f ", *(hz + i * ny + j));
      checksum += *(hz + i * ny + j) / (nx * ny);
    }
  //fprintf(stderr, "\nend   dump: %s\n", "hz");
  fprintf(stdout, "checksum = %lf\n", checksum);
}

static
void init_array (int tmax,
   int nx,
   int ny,
   float _fict_[ tmax])
{
  int i, j, idx = 0;
  for (i = 0; i < tmax; i++)
    _fict_[i] = (float) i;

  for (i = 0; i < nrow; i++)
  {
    for (j = 0; j < ny; j++)
      {
        idx = i + rank * nx / (ranksize - rest_helpers);
        *(ex + i * ny + j) = ((float) idx*(j+1)) / nx;
        *(ey + i * ny + j) = ((float) idx*(j+2)) / ny;
        *(hz + i * ny + j) = ((float) idx*(j+3)) / nx;
      }
  }
}

void error_handler(MPI_Comm *comm, int *err, ...)
{
    killed = 1;
    int num_failed = 0;
    MPI_Group group_failed;

    MPIX_Comm_failure_ack(comm_world);
    MPIX_Comm_failure_get_acked(comm_world, &group_failed);
    MPI_Group_size(group_failed, &num_failed);

    MPIX_Comm_shrink(comm_world, &comm_world);
    MPI_Comm_rank(comm_world, &rank);
    MPI_Comm_size(comm_world, &ranksize);

    if (num_failed > rest_helpers) {
        MPI_Abort(comm_world, 0);
    }

    rest_helpers -= num_failed;

    if (rank < ranksize - rest_helpers) {
        int startrow = (rank * NX) / (ranksize - rest_helpers);
        int lastrow = ((rank + 1) * NX) / (ranksize - rest_helpers) - 1;
        nrow = lastrow - startrow + 1;
        if (!ex) {
            ex = malloc(sizeof(*ex) * nrow * NY);
        } else {
            ex = realloc(ex, sizeof(*ex) * nrow * NY);
        }
        if (!ey) {
            ey = malloc(sizeof(*ey) * nrow * NY);
        } else {
            ey = realloc(ey, sizeof(*ey) * nrow * NY);
        }
        if (!hz) {
            hz = malloc(sizeof(*hz) * nrow * NY);
        } else {
            hz = realloc(hz, sizeof(*hz) * nrow * NY);
        }

        init_array (TMAX, NX, NY,
            _fict_);

        load();
    }

    load_iter();
    //printf("handle iter %d\n", t);

    MPI_Barrier(comm_world);
}

static
void kernel_fdtd_2d()
{
    int i, j;
    for(t = 0; t < TMAX; t++)
    {
        if (rank < ranksize - rest_helpers) {
            if (killed) {
                printf("continue after failure --- rank: %d  iter: %d\n", rank, t);
                killed = 0;
            }
            if (!rank)
            {
                for (j = 0; j < NY; j++)
                    *(ey + j) = _fict_[t];

                for (i = 1; i < nrow; i++)
                    for (j = 0; j < NY; j++)
                    {
                        *(ey + i * NY + j) = *(ey + i * NY + j) - 0.5f * (*(hz + i * NY + j) - *(hz + (i - 1) * NY + j));
                    }

                if (rank != ranksize - rest_helpers - 1)
                    MPI_Isend((hz + (nrow - 1) * NY), NY, MPI_FLOAT, rank + 1, t,
                            comm_world, &req[0]);
            }

            else
            {
                float temp[NY];
                MPI_Irecv(&temp[0], NY, MPI_FLOAT, rank - 1, t, comm_world,
                        &req[1]);
                if (rank != ranksize - rest_helpers - 1)
                {
                    MPI_Isend((hz + (nrow - 1) * NY), NY, MPI_FLOAT, rank + 1, t,
                            comm_world, &req[0]);
                }
                for (i = nrow - 1; i >= 0; i--)
                {
                    if (i == 0)
                    {
                        MPI_Status status;
                        MPI_Wait(&req[1], &status);
                    }

                    for (j = 0; j < NY; j++)
                    {
                        if (i == 0)
                        {
                            *(ey + i * NY + j) = *(ey + i * NY + j) - 0.5f * (*(hz + i * NY + j) - temp[j]);
                        }
                        else
                            *(ey + i * NY + j) = *(ey + i * NY + j) - 0.5f * (*(hz + i * NY + j) - *(hz + (i - 1) * NY + j));

                    }
                }
            }
            for (i = 0; i < nrow; i++)
                for (j = 1; j < NY; j++)
                    *(ex + i * NY + j) = *(ex + i * NY + j) - 0.5f * (*(hz + i * NY + j) - *(hz + i * NY + j - 1));

            if (rank != ranksize - rest_helpers - 1)
            {
                MPI_Status status;
                MPI_Wait(&req[0], &status);
            }

            float temp[NY];
            if (rank != ranksize - rest_helpers - 1)
            {
                MPI_Irecv(&temp[0], NY, MPI_FLOAT, rank + 1, t, comm_world,
                        &req[2]);
            }
            if (rank)
            {
                MPI_Isend(ey, NY, MPI_FLOAT, rank - 1, t, comm_world,
                        &req[3]);
            }
            for (i = 0; i < nrow; i++)
            {
                if (i == nrow - 1)
                {
                    if (rank == ranksize - rest_helpers - 1)
                    {
                        break;
                    }
                    else
                    {
                        MPI_Status status;
                        MPI_Wait(&req[2], &status);
                    }
                }
                for (j = 0; j < NY - 1; j++)
                {
                    if (i == nrow - 1)
                        *(hz + i * NY + j) = *(hz + i * NY + j) - 0.7f * (*(ex + i * NY + j + 1) - *(ex + i * NY + j) +
                                temp[j] - *(ey + i * NY + j));
                    else
                        *(hz + i * NY + j) = *(hz + i * NY + j) - 0.7f * (*(ex + i * NY + j + 1) - *(ex + i * NY + j) +
                                *(ey + (i + 1) * NY + j) - *(ey + i * NY + j));
                }
            }

            if (rank)
            {
                MPI_Status status;
                MPI_Wait(&req[3], &status);
            }

            if (t % 5 == 0) {
                dump();
                dump_iter();
            }
        }

        if (t > 5 && rank == ranksize - rest_helpers - 1 && rest_helpers) {
            printf("R.I.P. %d\n", rank);
            printf("killed on iter %d\n", t);
            raise(SIGKILL);
        }

        MPI_Barrier(comm_world);
    }
}


int main(int argc, char** argv)
{
  comm_world = MPI_COMM_WORLD;
  MPI_Init(&argc, &argv);
  MPI_Errhandler errh;
  MPI_Comm_create_errhandler(error_handler, &errh);
  MPI_Comm_set_errhandler(comm_world, errh);
  MPI_Comm_rank(comm_world, &rank);
  MPI_Comm_size(comm_world, &ranksize);
  int tmax = TMAX;
  int nx = NX;
  int ny = NY;
  MPI_Barrier(comm_world);
  double time_start = MPI_Wtime();

  // Give jobs only for active processes.
  if (rank < ranksize - HELPERS)
  {
    int startrow = (rank * nx) / (ranksize - rest_helpers);
    int lastrow = ((rank + 1) * nx) / (ranksize - rest_helpers) - 1;
    nrow = lastrow - startrow + 1;
    ex = malloc(sizeof(*ex) * nrow * ny);
    ey = malloc(sizeof(*ey) * nrow * ny);
    hz = malloc(sizeof(*hz) * nrow * ny);
  }


  if (rank < ranksize - HELPERS) {
      init_array (tmax, nx, ny,
         _fict_);
  }

  // bench_timer_start();
  kernel_fdtd_2d();

  MPI_Barrier(comm_world);

  float *result_ex = 0;
  float *result_ey = 0;
  float *result_hz = 0;

  if (!rank)
  {
      result_ex = malloc(nx * ny * sizeof(*result_ex));
      result_ey = malloc(nx * ny * sizeof(*result_ey));
      result_hz = malloc(nx * ny * sizeof(*result_hz));
  }

  int r = 0;
  int *recv_counts = malloc((ranksize - rest_helpers) * sizeof(*recv_counts));
  for (r = 0; r < (ranksize - rest_helpers); ++r)
  {
      int st = (r * nx) / (ranksize - rest_helpers);
      int lst = ((r + 1) * nx) / (ranksize - rest_helpers) - 1;
      int n = lst - st + 1;
      recv_counts[r] = n * ny;
  }

  int *displs = malloc((ranksize - rest_helpers) * sizeof(*displs));
  for (r = 0; r < (ranksize - rest_helpers); ++r)
  {
      if (!r)
      {
          displs[r] = 0;
      }
      else
      {
          displs[r] = displs[r - 1] +  recv_counts[r - 1];
      }
  }

  MPI_Gatherv(ex, nrow * ny, MPI_FLOAT, result_ex, recv_counts, displs, MPI_FLOAT, 0,
          comm_world);
  MPI_Gatherv(ey, nrow * ny, MPI_FLOAT, result_ey, recv_counts, displs, MPI_FLOAT, 0,
          comm_world);
  MPI_Gatherv(hz, nrow * ny, MPI_FLOAT, result_hz, recv_counts, displs, MPI_FLOAT, 0,
          comm_world);
  MPI_Barrier(comm_world);
  double time_end = MPI_Wtime();
  // bench_timer_stop();
  // bench_timer_print();

  if (!rank)
  {
    printf("\ntime = %lf\n", time_end - time_start);
    print_array(nx, ny, result_ex, result_ey, result_hz);
    free(result_ex);
    free(result_ey);
    free(result_hz);
  }
  MPI_Finalize();

  return 0;
}
