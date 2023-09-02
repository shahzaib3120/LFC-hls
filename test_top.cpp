#include <iostream>
#include <cmath>
#include <ctime>
#include <cstring>
#include <hls_stream.h>
#include <cstdlib>
#define AP_INT_MAX_W 8191
#include "ap_int.h"
#include "weights.hpp"
#include "bnn-library.h"
#include "data/memdata.h"
#include "data/config.h"
#include "activations.hpp"
#include "weights.hpp"
#include "activations.hpp"
#include "interpret.hpp"
#include "mvau.hpp"
#include "utils.hpp"
#include "fc_sw.hpp"
using namespace hls;
using namespace std;

#define MAX_IMAGES 1
const unsigned int inputPerSIMD = L0_MATRIXW / L0_SIMD;
// void LFC(stream<ap_uint<64>> &in,stream<ap_uint<64>> &out, unsigned int numReps);
// void LFC(stream<ap_uint<L0_SIMD * L0_INPUT_PRECISION>> &in, stream<ap_uint<L0_PE * L0_ACTIVATION_PRECISION>> &out, unsigned int numReps);
void LFC(stream<ap_uint<inputPerSIMD * L0_INPUT_PRECISION>> &in, stream<ap_uint<L1_PE * L1_ACTIVATION_PRECISION>> &out, unsigned int numReps);

int main()
{
	static ap_uint<L0_INPUT_PRECISION> IMAGE[MAX_IMAGES][L0_MATRIXW];
	static ap_uint<L0_ACTIVATION_PRECISION> TEST_INTER[MAX_IMAGES][L0_MATRIXH];
	static ap_uint<L1_ACTIVATION_PRECISION> TEST[MAX_IMAGES][L1_MATRIXH];
	stream<ap_uint<inputPerSIMD * L0_INPUT_PRECISION>> input_stream("input_stream");
#pragma HLS STREAM variable = input_stream depth = 128
	//	stream<ap_uint<L0_SIMD * L0_INPUT_PRECISION>> input_stream("input_stream");
	stream<ap_uint<L1_PE * L1_ACTIVATION_PRECISION>> output_stream("output_stream");
#pragma HLS STREAM variable = output_stream depth = 128
	auto capacity = input_stream.capacity();
	cout << "Input Stream Size = " << capacity << endl;

	// create image

	unsigned int counter = 0;

	// SIMD number of writes to stream, works only when SIMD=MW/SIMD or SIMD^2=MW, else needs StreamWidthAdjustment

	for (unsigned int n_img = 0; n_img < MAX_IMAGES; n_img++)
	{
		for (unsigned int simd = 0; simd < L0_SIMD; ++simd)
		{
			ap_uint<L0_INPUT_PRECISION * inputPerSIMD> simdInput = 0;
			for (unsigned int ips = 0; ips < inputPerSIMD; ++ips)
			{
				ap_uint<L0_INPUT_PRECISION> input = (ap_uint<L0_INPUT_PRECISION>)(counter);
				IMAGE[n_img][simd * inputPerSIMD + ips] = input;
				simdInput = simdInput >> L0_INPUT_PRECISION;
				simdInput(inputPerSIMD * L0_INPUT_PRECISION - 1, (inputPerSIMD - 1) * L0_INPUT_PRECISION) = input;
				printf("img[%d][%d]: %d\n", n_img, simd * inputPerSIMD + ips, input);
				counter++;
			}
			cout << "writing to stream = " << simdInput << endl;
			input_stream.write(simdInput);
			//			input_stream << simdInput;
		}
	}

	// Simple sequential write. Works only with SIMD = 1 otherwise wrong answer

	//	for(unsigned int n_img=0; n_img<MAX_IMAGES; n_img++)
	//	{
	//		for (unsigned int w = 0; w < L0_MATRIXW; ++w) {
	//			ap_uint<L0_INPUT_PRECISION> input = (ap_uint<L0_INPUT_PRECISION>)(counter);
	//			IMAGE[n_img][w]=input;
	//			input_stream.write(input);
	//			printf("img[%d][%d]: %d\n",n_img,w,input);
	////			cout << "img["<<n_img<<"]["<<w<<"] = " << input_stream.read() << endl;
	//			counter++;
	//		}
	//	}

	// software LFC

	// load weights

	//	template <MatrixW, MatrixH, SIMD, PE, TW>
	//	void loadFCWeights(ap_uint<L0_WIDTH> &weights[MatrixH][MatrixW], TW const &weights_in)
	static ap_uint<L0_WIDTH> W1[L0_MATRIXH][L0_MATRIXW];
	static ap_uint<L0_WIDTH> W2[L1_MATRIXH][L1_MATRIXW];
	loadFCWeights<L0_WIDTH, L0_MATRIXW, L0_MATRIXH, L0_SIMD, L0_PE>(W1, PARAM::weights_0);
	loadFCWeights<L1_WIDTH, L1_MATRIXW, L1_MATRIXH, L1_SIMD, L1_PE>(W2, PARAM::weights_1);
	// software inference

	//	template <int MAX_IMAGE,
	//	          int MATRIXW,
	//	          int MATRIXH,
	//	          typename TI,
	//	          typename TO,
	//	          typename TW>
	//	void fc_sw(TI const input[MAX_IMAGE][MATRIXW], TW const weights[MATRIXH][MATRIXW], TO output[MAX_IMAGE][MATRIXH])

	fc_sw<MAX_IMAGES, L0_MATRIXW, L0_MATRIXH, ap_uint<L0_INPUT_PRECISION>, ap_uint<L0_ACTIVATION_PRECISION>, ap_uint<L0_WIDTH>>(IMAGE, W1, TEST_INTER);
	fc_sw<MAX_IMAGES, L1_MATRIXW, L1_MATRIXH, ap_uint<L1_INPUT_PRECISION>, ap_uint<L1_ACTIVATION_PRECISION>, ap_uint<L1_WIDTH>>(TEST_INTER, W2, TEST);

	// print result

	//	template <MAX_IMAGES, MatrixH, TI>
	//	void printResult(TI &matrix[MAX_IMAGES][MatrixH])

	printResult<MAX_IMAGES, L1_MATRIXH, ap_uint<L1_ACTIVATION_PRECISION>>(TEST);

	LFC(input_stream, output_stream, MAX_IMAGES);

	bool error = false;

	unsigned int bit = 0;
	unsigned int output_reads = 0;
	while (!output_stream.empty())
	{
		auto output = output_stream.read();
		for (unsigned int i = 0; i < L1_PE; i++)
		{
			auto outElem = output((L1_ACTIVATION_PRECISION) * (i + 1) - 1, i * L1_ACTIVATION_PRECISION);
			cout << "output[" << bit << "] = " << outElem;
			cout << "\texpexted[" << bit << "] = " << TEST[0][bit] << endl;
			if (outElem != TEST[0][bit])
			{
				error = true;
			}
			bit++;
		}
		output_reads++;
	}
	cout << "Output Reads = " << output_reads << endl;
	if (error)
	{
		cout << "Test failed!" << endl;
		return 1;
	}
	else
	{
		cout << "Test passed!" << endl;
	}
	return 0;
}
