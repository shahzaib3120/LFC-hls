#ifndef UTILS_FC
#define UTILS_FC

#include <ap_int.h>
using namespace std;

// write a function to load weights
template <unsigned int WIDTH, unsigned int MatrixW, unsigned int MatrixH, unsigned int SIMD, unsigned int PE, typename TW>
void loadFCWeights(ap_uint<WIDTH> weights[MatrixH][MatrixW], TW const &weights_in)
{
    const int TX = MatrixW / SIMD;
    const int TY = MatrixH / PE;
    int kx = 0;
    int ky = 0;

    for (unsigned int oy = 0; oy < TY; oy++)
    {
        for (unsigned int pe = 0; pe < PE; pe++)
        {
            for (unsigned int ox = 0; ox < TX; ox++)
            {
                for (unsigned int simd = 0; simd < SIMD; simd++)
                {
                    weights[ky][kx] = weights_in.weights(oy * TX + ox)[pe][simd];
                    // cout << "TILE " << oy*TX + ox << " PE " << pe << " SIMD " << simd << endl;
                    if (kx == MatrixW - 1)
                    {
                        kx = 0;
                        ky++;
                    }
                    else
                    {
                        kx++;
                    }
                }
            }
        }
    }
}

template <unsigned int MAX_IMAGES, unsigned int MatrixH, typename TI>
void printResult(TI matrix[MAX_IMAGES][MatrixH])
{
    for (int i = 0; i < MAX_IMAGES; i++)
    {
        for (int j = 0; j < MatrixH; j++)
        {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
}
#endif
