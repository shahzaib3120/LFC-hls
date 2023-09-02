#include <hls_stream.h>
using namespace hls;
#include "ap_int.h"
#include "bnn-library.h"
#include "activations.hpp"
#include "weights.hpp"
#include "activations.hpp"
#include "interpret.hpp"
#include "mvau.hpp"
#include "data/memdata.h"
#include "data/config.h"
#include "fclayer.hpp"
const unsigned int inputPerSIMD = L0_MATRIXW / L0_SIMD;
// void LFC(stream<ap_uint<64>> &in, stream<ap_uint<64>> &out, unsigned int numReps)
void LFC(stream<ap_uint<inputPerSIMD * L0_INPUT_PRECISION>> &in, stream<ap_uint<L0_PE * L0_ACTIVATION_PRECISION>> &out, unsigned int numReps)
{
#pragma HLS DATAFLOW
  stream<ap_uint<L0_PE * L0_ACTIVATION_PRECISION>> fc1("FC1");
  StreamingFCLayer_Batch<L0_MATRIXW, L0_MATRIXH, L0_SIMD, L0_PE, L0_MMV, Slice<ap_uint<L0_INPUT_PRECISION>>, Slice<ap_uint<L0_ACTIVATION_PRECISION>>, Identity>(in, fc1, PARAM::weights_0, PassThroughActivation<ap_uint<L0_ACTIVATION_PRECISION>>(), numReps, ap_resource_dsp());
  StreamingFCLayer_Batch<L1_MATRIXW, L1_MATRIXH, L1_SIMD, L1_PE, L1_MMV, Slice<ap_uint<L1_INPUT_PRECISION>>, Slice<ap_uint<L1_ACTIVATION_PRECISION>>, Identity>(fc1, out, PARAM::weights_1, PassThroughActivation<ap_uint<L1_ACTIVATION_PRECISION>>(), numReps, ap_resource_dsp());

  /*
  template<
    unsigned MatrixW, unsigned MatrixH, unsigned SIMD, unsigned PE, unsigned MMV,
    typename TSrcI = Identity, typename TDstI = Identity, typename TWeightI = Identity,
    typename TI, typename TO, typename TW, typename TA, typename R
  >
  */

  //	unsigned const  InpPerImage = L0_MATRIXW / 64 * Slice<ap_uint<L0_INPUT_PRECISION>>::width;
  //	unsigned const  OutPerImage = L0_MATRIXH / L0_PE;

  //	WidthAdjustedInputStream <64, L0_SIMD*Slice<ap_uint<L0_INPUT_PRECISION>>::width, InpPerImage>  wa_in (in,  numReps);
  //	WidthAdjustedOutputStream<L0_PE*Slice<ap_uint<L0_ACTIVATION_PRECISION>>::width,  64, OutPerImage>  wa_out(out, numReps);

  // Matrix_Vector_Activate_Batch<L0_MATRIXW, L0_MATRIXH, L0_SIMD, L0_PE, L0_MMV, Slice<ap_uint<L0_INPUT_PRECISION>>, Slice<ap_uint<L0_ACTIVATION_PRECISION>>, Identity>(in, out, PARAM::weights_0, PassThroughActivation<ap_uint<L0_ACTIVATION_PRECISION>>(), numReps, ap_resource_dsp());
}
