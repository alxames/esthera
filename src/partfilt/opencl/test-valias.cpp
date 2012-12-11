/*
 * This file is part of the 'Esthera' bayesian estimation software toolkit.
 * Copyright (C) 2011-2012  Mehdi Chitchian and Alexander S. van Amesfoort
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/* test-valias.cpp
 * Author: Alexander S. van Amesfoort
 * Date: April 2012
 */
#include <cstdlib>
#include <cstring>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <string>
#include <vector>
#include <utility>
#include <iostream>
#include <fstream>
#include <stdexcept>

#include <boost/lexical_cast.hpp>

#define __CL_ENABLE_EXCEPTIONS
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

struct progArgs {
	cl_device_type devTypeMask;
	size_t nrounds;
	size_t nparticles;
	size_t nfilters;
};

using namespace std;

static string readFile(const string& filename) {
	int fd = open(filename.c_str(), O_RDONLY);
	if (fd == -1) throw runtime_error("Failed to open OpenCL kernel source file");

	struct stat statbuf;
	int err = fstat(fd, &statbuf);
	if (err == -1) { close(fd); throw runtime_error("Failed to retrieve OpenCL kernel source file size"); }

	char* data = (char*)malloc(statbuf.st_size + 1);
	if (data == NULL) { close(fd); throw runtime_error("Failed to allocate memory to read OpenCL kernel source file"); }

	size_t ntotal = 0;
	while (ntotal < statbuf.st_size) {
		ssize_t nread = read(fd, data + ntotal, statbuf.st_size - ntotal);
		if (nread <= 0) { free(data); close(fd); throw runtime_error("Failed to read OpenCL kernel source file data"); }
		ntotal += (size_t)nread;
	}
	data[ntotal] = '\0';

	int err2 = close(fd);
	if (err2 == -1) cerr << "Warning: failed to close OpenCL kernel source file" << endl;

	string sdata(data);
	free(data);
	return sdata;
}

static void main2(struct progArgs& args) {
	vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	if (platforms.empty()) throw runtime_error("No OpenCL platforms detected");

	vector<cl::Device> devices;
	platforms[0].getDevices(args.devTypeMask, &devices);
	if (devices.empty()) throw runtime_error("No OpenCL devices detected in 1st (masked) platform"); // getDevices() may throw already

	devices.resize(1);
	cl_context_properties ctxProps[1] = {0};
	cl::Context context(cl::Context(devices, ctxProps, 0, 0, 0));

	cl_command_queue_properties cqProps = 0;
	cl::CommandQueue cq(context, devices[0], cqProps, 0);

	string srcFilename("resampling-valias.cl");
	string srcText(readFile(srcFilename));

	cl::Program::Sources progSources;
	progSources.push_back(make_pair(srcText.c_str(), srcText.size()));
	cl::Program program(context, progSources, 0);

	string kBuildOptions("-DDEBUG=1 -DTEST_VALIAS -cl-opt-disable -g"); // -g works on AMD APP
	try {
		program.build(devices, kBuildOptions.c_str());
	} catch (cl::Error& clerr) {
		if (clerr.err() == CL_BUILD_PROGRAM_FAILURE) {
			string buildLog(program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0], 0));
			if (!buildLog.empty()) cout << "OpenCL compiler diagnostics:" << endl << buildLog << endl;
		}
		throw;
	}
	// Also print build warnings if any.
	string buildLog(program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0], 0));
	if (!buildLog.empty()) cout << "OpenCL compiler diagnostics:" << endl << buildLog << endl;

	string kernelFuncName("resampling_valias");
	cl::Kernel kernel(program, kernelFuncName.c_str(), 0);


	size_t nvals = args.nparticles * args.nfilters;
	size_t size  = args.nparticles * args.nfilters * sizeof(cl_float);
	size_t size4 = args.nparticles * args.nfilters * sizeof(cl_float4);

	// __global
	// Irrel for our tests; set to 0.
	cl::Buffer angles1    (context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, size4);
	cl::Buffer angles2    (context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, size);
	cl::Buffer pos        (context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, size4);
	cl::Buffer tmp_angles1(context, CL_MEM_READ_ONLY  | CL_MEM_ALLOC_HOST_PTR, size4);
	cl::Buffer tmp_angles2(context, CL_MEM_READ_ONLY  | CL_MEM_ALLOC_HOST_PTR, size);
	cl::Buffer tmp_pos    (context, CL_MEM_READ_ONLY  | CL_MEM_ALLOC_HOST_PTR, size4);
	// Relevant, test various weight vectors
	cl::Buffer weights    (context, CL_MEM_READ_ONLY  | CL_MEM_ALLOC_HOST_PTR, size); // true resampling would use CL_MEM_READ_WRITE
	// Relevant, test various rng numbers (always deterministic for testing)
	size_t nrand_nrs = (2*args.nparticles+1) * args.nfilters; // Vose' alias method needs 2 nrs per particle and 1 nr per group to see if to resample.
	size_t nrand_size = nrand_nrs * sizeof(cl_float);
	cl::Buffer random_nrs (context, CL_MEM_READ_ONLY  | CL_MEM_ALLOC_HOST_PTR, nrand_size);
	// imm
	int num_particles = (int)args.nparticles; // redundant: always 1 thread per particle, so equals get_local_size(0)
	float resampling_bound = 1.0f; // always resample everything when testing
//	int rand_offset = 0; // redundant, but an impl-dep pain to use sub-buffers

	cl_uint aidx = 0;
	kernel.setArg(aidx++, angles1);
	kernel.setArg(aidx++, angles2);
	kernel.setArg(aidx++, pos);
	kernel.setArg(aidx++, tmp_angles1);
	kernel.setArg(aidx++, tmp_angles2);
	kernel.setArg(aidx++, tmp_pos);
	kernel.setArg(aidx++, weights);
	kernel.setArg(aidx++, random_nrs);
	kernel.setArg(aidx++, num_particles);
	kernel.setArg(aidx++, resampling_bound);
//	kernel.setArg(aidx++, rand_offset);

	// __local
	kernel.setArg(aidx++, num_particles * sizeof(cl_float), NULL);
	kernel.setArg(aidx++, num_particles * sizeof(cl_uint), NULL);
	kernel.setArg(aidx++, num_particles * sizeof(cl_uint), NULL); // actually: 'num_particles * max(sizeof(cl_uint), sizeof(cl_float))'

	// Exec config
	cl::NDRange offset(0);
	cl::NDRange global_size(nvals);
	cl::NDRange local_size(args.nparticles);

	// map and init and unmap irrelevant buffers
	srand(0x11111111U);
	cl_float4* h_angles1    = (cl_float4*)cq.enqueueMapBuffer(angles1,    CL_TRUE, CL_MAP_WRITE, 0, size4);
	cl_float*  h_angles2    = (cl_float*) cq.enqueueMapBuffer(angles2,    CL_TRUE, CL_MAP_WRITE, 0, size);
	cl_float4* h_pos        = (cl_float4*)cq.enqueueMapBuffer(pos,        CL_TRUE, CL_MAP_WRITE, 0, size4);
	memset(h_angles1, 0, size4);
	memset(h_angles2, 0, size);
	memset(h_pos,     0, size4);
	cq.enqueueUnmapMemObject(angles1, h_angles1);
	cq.enqueueUnmapMemObject(angles2, h_angles2);
	cq.enqueueUnmapMemObject(pos    , h_pos);

for (size_t i = 0; i < args.nrounds; i++) {
	// map to init
	cl_float*  h_weights    = (cl_float*) cq.enqueueMapBuffer(weights,    CL_TRUE, CL_MAP_WRITE, 0, size);
	cl_float*  h_random_nrs = (cl_float*) cq.enqueueMapBuffer(random_nrs, CL_TRUE, CL_MAP_WRITE, 0, nrand_size);

	// init
	for (size_t i = 0; i < nvals; i++) {
		int rn = rand();
		h_weights[i] = (float)(rn / ((double)RAND_MAX+1)); // to [0.0f, 1.0f)
	}

	for (size_t i = 0; i < nrand_nrs; i++) {
		int rn = rand();
		h_random_nrs[i] = (float)(rn / ((double)RAND_MAX+1)); // to [0.0f, 1.0f)
	}

	// On the CPU it's better to do an unmap and for testing here, only care about that.
	cq.enqueueUnmapMemObject(weights,    h_weights);
	cq.enqueueUnmapMemObject(random_nrs, h_random_nrs);

	// run
	cq.enqueueNDRangeKernel(kernel, offset, global_size, local_size);
	cq.finish();

	//printf("-----------------------------------------\n"); // separate round printing
}

	// to check results
	//cq.enqueueMapBuffer(tmp_angles1, CL_TRUE, CL_MAP_READ, 0, size4);
	//cq.enqueueMapBuffer(tmp_angles2, CL_TRUE, CL_MAP_READ, 0, size);
	//cq.enqueueMapBuffer(tmp_pos,     CL_TRUE, CL_MAP_READ, 0, size4);
}

static void parseArgs(vector<string>& cmdLineArgs, struct progArgs& args) {
	// Default values
	args.devTypeMask = CL_DEVICE_TYPE_ALL; // still excludes CL_DEVICE_TYPE_CUSTOM
	args.nrounds = 2;
	args.nparticles = 256;
	args.nfilters = 16;

	for (unsigned i = 1; i < cmdLineArgs.size(); i++) {
		if (cmdLineArgs[i].find("--device=") == 0) {
			string devType(cmdLineArgs[i].substr(strlen("--device=")));
			if      (devType == "CPU") args.devTypeMask = CL_DEVICE_TYPE_CPU;
			else if (devType == "GPU") args.devTypeMask = CL_DEVICE_TYPE_GPU;
			else throw runtime_error("Unknown device type argument");
		} else if (cmdLineArgs[i].find("--rounds=") == 0) {
			string rounds(cmdLineArgs[i].substr(strlen("--rounds=")));
			args.nrounds = boost::lexical_cast<size_t>(rounds);
		} else if (cmdLineArgs[i].find("--N=") == 0) {
			string dims(cmdLineArgs[i].substr(strlen("--N=")));
			args.nparticles = boost::lexical_cast<size_t>(dims);
			if (args.nparticles > 1024) throw runtime_error("--N cannot be > 1024");
		} else if (cmdLineArgs[i].find("--m=") == 0) {
			string dims(cmdLineArgs[i].substr(strlen("--m=")));
			args.nfilters = boost::lexical_cast<size_t>(dims);
			if (args.nfilters > 1024*1024) throw runtime_error("--m cannot be > 1024*1024");
		} else {
			cerr << "Usage: " << cmdLineArgs[0] << " [--device=CPU|GPU] [--rounds=2] [--N=256] [--m=16]" << endl;
			cerr << "\t--rounds\tNumber of rounds to execute" << endl;
			cerr << "\t--N\t\tNumber of particles per sub-filter (<= 1024)" << endl;
			cerr << "\t--m\t\tNumber of sub-filters (<= 1024*1024)" << endl;
			throw runtime_error("Unknown command line argument");
		}
	}
}

static vector<string> cstrToVector(char* cstrings[], size_t nstrings) {
	vector<string> res;
	for (size_t i = 0; i < nstrings; i++) {
		res.push_back(cstrings[i]);
	}
	return res;
}

int main(int argc, char* argv[]) {
	struct progArgs progArgs;

	vector<string> cmdLineArgs(cstrToVector(argv, argc));
	parseArgs(cmdLineArgs, progArgs);

	try {
		main2(progArgs);
	} catch (cl::Error& clerr) {
		cerr << "CL Error: " << clerr.what() << ": " << clerr.err() <<  endl;
	} catch (exception& exc) {
		cerr << "Exception: " << exc.what() << endl;
	} catch (...) {
		cerr << "Exception: unknown error" << endl;
	}

	return 0;
}

