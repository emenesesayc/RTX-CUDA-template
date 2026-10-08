#pragma once

#include <unistd.h>
#include <string>
#include <time.h>
#include <getopt.h>
#include <iostream>
#include <fstream>
#include <cxxopts.hpp>


struct CmdArgs {
    int n, alg, steps, reps, dev, nt, seed, check, save_time, save_power;
    std::string time_file, power_file;
};

struct Results {
    float output;
    float time;
    int mem;
    int power;
};


CmdArgs get_args(int argc, char *argv[]) {
    cxxopts::Options options("rtxcuda", "Template for rtx-compute");
    options.add_options()
        ("h,help", "Print help")
	("n", "Problem size", cxxopts::value<int>())
	("a,alg", "Algorithm: 1) cuda, 2) cub, 3) thrust, 4) rtx", cxxopts::value<int>())
	("seed", "Seed for PRNG", cxxopts::value<int>())
	("reps", "Number of repetitions for the avg time", cxxopts::value<int>()->default_value("1"))
	("nt", "Number of CPU threads", cxxopts::value<int>()->default_value("1"))
	("check", "Check correctness")
	("save-time", "Save time measurements", cxxopts::value<std::string>()->default_value(""))
	("save-power", "Save power measurements", cxxopts::value<std::string>()->default_value(""))
	("dev", "GPU device id", cxxopts::value<int>()->default_value("0"));

    auto result = options.parse(argc, argv);

    if (result.count("help") || !result.count("n") || !result.count("alg")) {
        std::cout << options.help({""}) << std::endl;
	std::exit(0);
    }

    CmdArgs args;
    args.n = result["n"].as<int>();
    args.alg = result["alg"].as<int>();
    args.seed = result.count("seed") ? result["seed"].as<int>() : time(0);
    args.reps = result["reps"].as<int>();
    args.nt = result["nt"].as<int>();
    args.dev = result["dev"].as<int>();
    args.check = result.count("check");
    args.save_time = result.count("save-time");
    args.time_file = result["save-time"].as<std::string>();
    args.save_power = result.count("save-power");
    args.power_file = result["save-power"].as<std::string>();
	


    printf( "Params:\n"
            "   reps = %i\n"
            "   seed = %i\n"
            "   dev  = %i\n"
            AC_GREEN "   n     = %i (~%f GB, float)\n" AC_RESET
            //AC_GREEN "   q    = %i (~%f GB, int2)\n" AC_RESET
            AC_GREEN "   steps = %i\n" AC_RESET
            "   nt   = %i CPU threads\n"
            "   alg  = %i (%s)\n\n",
            args.reps, args.seed, args.dev, args.n, sizeof(float)*args.n/1e9, args.steps,
            args.nt, args.alg, algStr[args.alg]);

    return args;
}

bool is_equal(float a, float b) {
    float epsilon = 1e-4f;
    return abs(a - b) < epsilon;
}


void print_gpu_specs(int dev) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

    // Use cudaDeviceGetAttribute to get deprecated properties
    int memoryClockRateKhz, memoryBusWidthBits;
    cudaDeviceGetAttribute(&memoryClockRateKhz, cudaDevAttrMemoryClockRate, dev);
    cudaDeviceGetAttribute(&memoryBusWidthBits, cudaDevAttrGlobalMemoryBusWidth, dev);

    printf("Device Number: %d\n", dev);
    printf("  Device name:                  %s\n", prop.name);
    printf("  Multiprocessor Count:         %d\n", prop.multiProcessorCount);
    printf("  Concurrent Kernels:           %d\n", prop.concurrentKernels);

    // Print the memory clock rate, converting kHz to MHz
    printf("  Memory Clock Rate (MHz):      %d\n", memoryClockRateKhz / 1000);
    // Print the memory bus width
    printf("  Memory Bus Width (bits):      %d\n", memoryBusWidthBits);

    // Calculate and print the peak memory bandwidth
    double peak_bandwidth = 2.0 * memoryClockRateKhz * memoryBusWidthBits / 8.0 / 1.0e6;
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n", peak_bandwidth);
}


void write_results(CmdArgs args, Results results) {
    if (!args.save_time) return;
    std::string filename;
    if (args.time_file.empty())
        filename = std::string("../results/data.csv");
    else
        filename = args.time_file;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, args.dev);
    //char *device = prop.name;
    //if (alg == ALG_CPU_BASE || alg == ALG_CPU_HRMQ) {
    //    strcpy(device, "CPU ");
    //    char hostname[50];
    //    gethostname(hostname, 50);
    //    strcat(device, hostname);
    //}

    FILE *fp;
    fp = fopen(filename.c_str(), "a");

    fprintf(fp, 
            "%s,%s,%i,%i," //args
            "%f\n", // results
            args.dev,
            algStr[args.alg],
            args.reps,
            args.n,
            results.time);
    fclose(fp);
}
