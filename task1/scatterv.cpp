#include <iostream>
#include <mpi.h>

int
main(int argc, char **argv) {
    int wrank, wsize;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &wsize);
    MPI_Comm_rank(MPI_COMM_WORLD, &wrank);

    if (!wrank) {
        for (int i = 15; i >= 1; --i) {
            int *buf = (int *) malloc(i * sizeof(*buf));
            for (int j = 0; j < i; ++j) {
                buf[j] = j + 1;
            }

            int dst = (i - wrank) % 4 == 0 ? wrank + 4 : wrank + 1;
            MPI_Send(&i, 1, MPI_INT, dst, 1, MPI_COMM_WORLD);
            MPI_Send(buf, i, MPI_INT, dst, 1, MPI_COMM_WORLD);
        }

        std::cout << "Process 0 sent all messages and finished." << std::endl;
        MPI_Finalize();
    } else {
        int dst = 0;
        int src = 0;

        if (wrank > 3) {
            src = wrank - 4;
        } else {
            src = wrank - 1;
        }

        MPI_Status status;
        while (1) {
            MPI_Recv(&dst, 1, MPI_INT, src, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            int *buf = (int *) malloc(dst * sizeof(*buf));
            MPI_Recv(buf, dst, MPI_INT, src, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (dst == wrank) {
                std::cout << "Process "<< wrank << " message: ";
                for (int i = 0; i < dst; ++i) {
                    std::cout << buf[i] << " ";
                }
                std::cout << std::endl;

                MPI_Finalize();
                break;
            } else {
                int neighbor = (dst - wrank) % 4 == 0 ? wrank + 4 : wrank + 1;
                MPI_Send(&dst, 1, MPI_INT, neighbor, 1, MPI_COMM_WORLD);
                MPI_Send(buf, dst, MPI_INT, neighbor, 1, MPI_COMM_WORLD);
            }
            free(buf);
        }
    }

    return 0;
}
