// Resource-only inspection using the exact ABI library. No evaluation launch.
#include "ToucanGPUSim.h"
#include "SimEval.h"
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <cuda_runtime.h>
static void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
int main(int argc,char**argv)try{
 if(argc!=2)return 2;
 toucanGPUSim::ToucanSimulator sim{};
 if(sim.init(0,std::string(argv[1])+"/GPUSimDesign.bin",std::string(argv[1])+"/GPUSimIOSymbols.bin",false,false))return 3;
 cudaDeviceProp prop{};check(cudaGetDeviceProperties(&prop,0));
 size_t dynamic=0,parts=0;for(const auto&p:sim.design.parts)dynamic=std::max(dynamic,size_t(p.valuePoolSize));
 for(const auto&r:sim.design.regionPartitionIds)parts=std::max(parts,r.size());
 const void* kernels[]={reinterpret_cast<const void*>(evalSingleCycle),reinterpret_cast<const void*>(evalFreeRunningNCycles),reinterpret_cast<const void*>(evalFreeRunningNCycles_Large)};
 const char* names[]={"evalSingleCycle","evalFreeRunningNCycles","evalFreeRunningNCycles_Large"};
 cudaFuncAttributes attrs[3]{};int threads=prop.maxThreadsPerBlock,blocks[3]={};
 for(int i=0;i<3;++i){check(cudaFuncGetAttributes(attrs+i,kernels[i]));threads=std::min(threads,attrs[i].maxThreadsPerBlock);}
 bool fit=false;for(;threads>=64;threads>>=1){fit=true;for(int i=0;i<3;++i){check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(blocks+i,kernels[i],threads,dynamic));fit &= blocks[i]>0;}if(fit)break;}
 if(!fit)throw std::runtime_error("No residency at actual dynamic shared size");
 int grid=std::min(parts,size_t(std::min(blocks[1],blocks[2])*prop.multiProcessorCount));
 std::cout<<"RESOURCE_JSON={\"device\":\""<<prop.name<<"\",\"SMs\":"<<prop.multiProcessorCount<<",\"threads\":"<<threads<<",\"dynamic_shared_bytes\":"<<dynamic<<",\"max_round_partitions\":"<<parts<<",\"cooperative_grid\":"<<grid<<",\"batch_kernel_by_frozen_dispatch\":\""<<(parts>size_t(grid)?names[2]:names[1])<<"\",\"device_max_shared_optin\":"<<prop.sharedMemPerBlockOptin<<",\"device_shared_per_SM\":"<<prop.sharedMemPerMultiprocessor<<",\"kernels\":[";
 for(int i=0;i<3;++i){if(i)std::cout<<",";std::cout<<"{\"name\":\""<<names[i]<<"\",\"registers\":"<<attrs[i].numRegs<<",\"static_shared_bytes\":"<<attrs[i].sharedSizeBytes<<",\"local_bytes\":"<<attrs[i].localSizeBytes<<",\"max_dynamic_shared_bytes\":"<<attrs[i].maxDynamicSharedSizeBytes<<",\"resident_blocks_per_SM\":"<<blocks[i]<<"}";}
 std::cout<<"],\"evaluation_launched\":false}\n";return 0;
}catch(const std::exception&e){std::cerr<<"RESOURCE_PROBE_FAIL "<<e.what()<<"\n";return 1;}
