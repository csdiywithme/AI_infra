// Infrastructure only: includes the user's preserved kernels without changing them.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include "src/cuda/common.h"
#include "contract.h"

#define STAGES 2
namespace baseline {
#define main unused_baseline_main
#include "src/cuda/m4_gemm/baseline.cu"
#undef main
}
namespace candidate {
#define main unused_candidate_main
#include "src/cuda/m4_gemm/candidate.cu"
#undef main
}
#undef STAGES

#define BLAS_CHECK(call) do { auto s = (call); if (s != CUBLAS_STATUS_SUCCESS) { \
    std::fprintf(stderr,"cuBLAS error %d at line %d\n",int(s),__LINE__);std::exit(2); }} while(0)
#define DRIVER_CHECK(call) do { auto s = (call); if (s != CUDA_SUCCESS) { \
    std::fprintf(stderr,"CUDA driver error %d at line %d\n",int(s),__LINE__);std::exit(3); }} while(0)

static_assert(baseline::BM==128 && baseline::BN==256 && baseline::BK==64 &&
              baseline::NSTAGE==2 && baseline::THREADS==160, "Frozen baseline ABI changed");
static_assert(candidate::BM==128 && candidate::BN==256 && candidate::BK==64 &&
              candidate::NSTAGE==2 && candidate::THREADS==160, "Candidate changed frozen launch ABI");
constexpr size_t kSmem = 2 * (128+256) * 64 * 2 + 1024;
constexpr const char* kImpls[] = {"baseline","candidate"};

struct Case {
    ContractShape shape;
    int sm_count, grid;
    size_t na,nb,nd;
    __nv_bfloat16 *a=nullptr,*b=nullptr;
    float *d=nullptr,*reference=nullptr;
    std::vector<__nv_bfloat16> initial_a,initial_b;
    CUtensorMap ta={},tb={};
    cublasHandle_t handle;
    Case(ContractShape sh,int sms,cublasHandle_t h):shape(sh),sm_count(sms),handle(h) {
        na=size_t(sh.m)*sh.k;nb=size_t(sh.n)*sh.k;nd=size_t(sh.m)*sh.n;
        grid=std::min((sh.m/128)*(sh.n/256),sms*2);
        CUDA_CHECK(cudaMalloc(&a,na*sizeof(*a)));CUDA_CHECK(cudaMalloc(&b,nb*sizeof(*b)));
        CUDA_CHECK(cudaMalloc(&d,nd*sizeof(*d)));CUDA_CHECK(cudaMalloc(&reference,nd*sizeof(*reference)));
        const cuuint64_t da[]={cuuint64_t(sh.k),cuuint64_t(sh.m)};
        const cuuint64_t db[]={cuuint64_t(sh.k),cuuint64_t(sh.n)};
        const cuuint64_t strides[]={cuuint64_t(sh.k)*2};
        const cuuint32_t boxa[]={64,128},boxb[]={64,256},elem[]={1,1};
        auto encode=[&](CUtensorMap* map,void* ptr,const cuuint64_t* dims,const cuuint32_t* box) {
            DRIVER_CHECK(cuTensorMapEncodeTiled(map,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,ptr,dims,
                strides,box,elem,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,
                CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
        };
        encode(&ta,a,da,boxa);encode(&tb,b,db,boxb);
    }
    ~Case(){cudaFree(a);cudaFree(b);cudaFree(d);cudaFree(reference);}
    void launch(int impl) {
        if(impl==0) baseline::gemm_pipeline<<<grid,160,kSmem>>>(a,b,d,shape.m,shape.n,shape.k,ta,tb);
        else candidate::gemm_pipeline<<<grid,160,kSmem>>>(a,b,d,shape.m,shape.n,shape.k,ta,tb);
    }
    void oracle() {
        const float alpha=1.f,beta=0.f;
        BLAS_CHECK(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,shape.n,shape.m,shape.k,&alpha,
            b,CUDA_R_16BF,shape.k,a,CUDA_R_16BF,shape.k,&beta,reference,CUDA_R_32F,shape.n,
            CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    void fill(unsigned seed,int kind) {
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> ints(-3,3);
        std::uniform_real_distribution<float> fractions(-1.f,1.f);
        std::vector<__nv_bfloat16> host(std::max(na,nb));
        auto upload=[&](__nv_bfloat16* dst,size_t count,bool is_a) {
            for(size_t i=0;i<count;++i) {
                float v;
                if(kind==0) v=float(ints(rng));
                else if(kind==1) v=fractions(rng);
                else {
                    size_t row=i/shape.k,col=i%shape.k;
                    v=is_a ? (col==size_t(shape.k-1)?float(int((row+seed)%7)-3):0.f)
                           : float(int((row+seed)%11)-5)*(col%13==seed%13?1.f:0.125f);
                }
                host[i]=__float2bfloat16(v);
            }
            CUDA_CHECK(cudaMemcpy(dst,host.data(),count*sizeof(*dst),cudaMemcpyHostToDevice));
            if(is_a)initial_a.assign(host.begin(),host.begin()+count);
            else initial_b.assign(host.begin(),host.begin()+count);
        };
        upload(a,na,true);upload(b,nb,false);
    }
    bool input_integrity(const char* phase) {
        std::vector<__nv_bfloat16> actual(std::max(na,nb));
        auto count_bad=[&](__nv_bfloat16* ptr,size_t count,const std::vector<__nv_bfloat16>& original) {
            CUDA_CHECK(cudaMemcpy(actual.data(),ptr,count*sizeof(*ptr),cudaMemcpyDeviceToHost));
            size_t bad=0;
            for(size_t i=0;i<count;++i)bad+=std::memcmp(&actual[i],&original[i],sizeof(*ptr))!=0;
            return bad;
        };
        const size_t bad_a=count_bad(a,na,initial_a),bad_b=count_bad(b,nb,initial_b);
        bool passed=bad_a==0&&bad_b==0;
        std::printf("{\"event\":\"input_integrity\",\"shape_id\":\"%dx%dx%d\","
                    "\"phase\":\"%s\",\"bad_a\":%zu,\"bad_b\":%zu,\"passed\":%s}\n",
                    shape.m,shape.n,shape.k,phase,bad_a,bad_b,passed?"true":"false");
        std::fflush(stdout);
        return passed;
    }
    bool timing_validation(int block,int impl,const std::vector<float>& gold) {
        std::vector<float> actual(nd);
        CUDA_CHECK(cudaMemcpy(actual.data(),d,nd*sizeof(float),cudaMemcpyDeviceToHost));
        size_t bad=0,nonfinite=0;
        for(size_t i=0;i<nd;++i) {
            double got=actual[i],want=gold[i];
            if(!std::isfinite(got)||!std::isfinite(want)){++bad;++nonfinite;continue;}
            bad+=std::abs(got-want)>kAtol+kRtol*std::abs(want);
        }
        bool passed=bad==0;
        std::printf("{\"event\":\"timing_validation\",\"shape_id\":\"%dx%dx%d\","
                    "\"block\":%d,\"impl\":\"%s\",\"bad\":%zu,\"nonfinite\":%zu,"
                    "\"elements\":%zu,\"passed\":%s}\n",shape.m,shape.n,shape.k,block,
                    kImpls[impl],bad,nonfinite,nd,passed?"true":"false");
        std::fflush(stdout);
        return passed;
    }
    bool validate(int shape_id,const char* phase) {
        const char* names[]={"integer_exact","random_bf16","structured"};
        bool all=true;
        for(int si=0;si<kSeedCount;++si) for(int kind=0;kind<3;++kind) {
            unsigned seed=kSeeds[si];fill(seed,kind);oracle();
            std::vector<float> gold(nd),actual(nd);
            CUDA_CHECK(cudaMemcpy(gold.data(),reference,nd*sizeof(float),cudaMemcpyDeviceToHost));
            size_t gold_nonfinite=0;for(float v:gold)gold_nonfinite+=!std::isfinite(v);
            for(int impl=0;impl<2;++impl) {
                CUDA_CHECK(cudaMemset(d,0xff,nd*sizeof(float)));
                launch(impl);CUDA_CHECK_KERNEL();
                CUDA_CHECK(cudaMemcpy(actual.data(),d,nd*sizeof(float),cudaMemcpyDeviceToHost));
                size_t bad=0,nonfinite=0;double max_abs=0;
                for(size_t i=0;i<nd;++i) {
                    double got=actual[i],want=gold[i];
                    if(!std::isfinite(got)||!std::isfinite(want)){++bad;++nonfinite;continue;}
                    double diff=std::abs(got-want),bound=kind==0&&kIntegerExact ? 0. : kAtol+kRtol*std::abs(want);
                    max_abs=std::max(max_abs,diff);bad+=diff>bound;
                }
                bool passed=bad==0&&gold_nonfinite==0;all&=passed;
                std::printf("{\"event\":\"validation\",\"phase\":\"%s\",\"shape_id\":\"%dx%dx%d\","
                    "\"impl\":\"%s\",\"seed\":%u,\"data\":\"%s\",\"elements\":%zu,"
                    "\"bad\":%zu,\"nonfinite\":%zu,\"reference_nonfinite\":%zu,"
                    "\"max_abs_error\":%.17g,\"passed\":%s}\n",
                    phase,shape.m,shape.n,shape.k,kImpls[impl],seed,names[kind],nd,bad,nonfinite,gold_nonfinite,
                    max_abs,passed?"true":"false");
                std::fflush(stdout);
                if(!passed)return false;
            }
        }
        return all;
    }
    double timed_slot(int impl) {
        for(int i=0;i<kWarmup;++i)launch(impl);
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEvent_t start,end;CUDA_CHECK(cudaEventCreate(&start));CUDA_CHECK(cudaEventCreate(&end));
        CUDA_CHECK(cudaEventRecord(start));
        for(int i=0;i<kIterations;++i)launch(impl);
        CUDA_CHECK(cudaEventRecord(end));CUDA_CHECK(cudaEventSynchronize(end));
        CUDA_CHECK(cudaGetLastError());
        float total;CUDA_CHECK(cudaEventElapsedTime(&total,start,end));
        CUDA_CHECK(cudaEventDestroy(start));CUDA_CHECK(cudaEventDestroy(end));
        double result=double(total)/kIterations;
        if(!std::isfinite(result)||result<=0){std::fprintf(stderr,"Invalid event timing\n");std::exit(4);}
        return result;
    }
    void timing(int shape_id) {
        fill(kSeeds[kSeedCount-1],1);
        oracle();
        std::vector<float> gold(nd);
        CUDA_CHECK(cudaMemcpy(gold.data(),reference,nd*sizeof(float),cudaMemcpyDeviceToHost));
        if(!input_integrity("timing_before"))std::exit(12);
        // Same A/B/output allocations, default stream, no explicit cache flush.
        // Warm both implementations before the randomized balanced schedule.
        for(int i=0;i<100;++i){launch(0);launch(1);}CUDA_CHECK_KERNEL();
        std::vector<int> patterns(kBlocks,0);
        for(int i=kBlocks/2;i<kBlocks;++i)patterns[i]=1;
        std::mt19937 order_rng(kOrderSeed+unsigned(shape_id));
        std::shuffle(patterns.begin(),patterns.end(),order_rng);
        for(int block=0;block<kBlocks;++block) {
            const char* pattern=patterns[block]?"BAAB":"ABBA";
            for(int slot=0;slot<4;++slot) {
                int impl=pattern[slot]=='A'?0:1;
                double ms=timed_slot(impl);
                std::printf("{\"event\":\"timing\",\"shape_id\":\"%dx%dx%d\",\"block\":%d,"
                    "\"pattern\":\"%s\",\"slot\":%d,\"impl\":\"%s\",\"latency_ms\":%.17g,"
                    "\"iterations\":%d}\n",shape.m,shape.n,shape.k,block,pattern,slot,kImpls[impl],ms,kIterations);
                std::fflush(stdout);
            }
            if(!timing_validation(block,pattern[3]=='A'?0:1,gold))std::exit(13);
        }
        if(!input_integrity("timing_after"))std::exit(14);
    }
};

int main(int argc,char** argv) {
    const std::string mode=argc>1?argv[1]:"check";
    if(mode!="check"&&mode!="bench"&&mode!="profile-target")return 64;
    cudaDeviceProp props{};CUDA_CHECK(cudaGetDeviceProperties(&props,0));
    if(props.major!=10||props.minor!=3||std::string(props.name).find("B300")==std::string::npos) {
        std::fprintf(stderr,"Expected B300 sm_103, got %s (%d,%d)\n",props.name,props.major,props.minor);return 65;
    }
    std::printf("{\"event\":\"environment\",\"gpu_name\":\"%s\",\"sm_count\":%d,"
        "\"cc_major\":%d,\"cc_minor\":%d}\n",props.name,props.multiProcessorCount,props.major,props.minor);
    std::fflush(stdout);
    DRIVER_CHECK(cuInit(0));
    CUDA_CHECK(cudaFuncSetAttribute(baseline::gemm_pipeline,cudaFuncAttributeMaxDynamicSharedMemorySize,kSmem));
    CUDA_CHECK(cudaFuncSetAttribute(candidate::gemm_pipeline,cudaFuncAttributeMaxDynamicSharedMemorySize,kSmem));
    cublasHandle_t handle;BLAS_CHECK(cublasCreate(&handle));
    BLAS_CHECK(cublasSetMathMode(handle,CUBLAS_DEFAULT_MATH));
    if(mode=="profile-target") {
        int id=0;while(id<kShapeCount&&!kShapes[id].acceptance)++id;
        if(id==kShapeCount)return 66;
        Case c(kShapes[id],props.multiProcessorCount,handle);c.fill(kSeeds[0],1);
        // NCU launch-skip skips these warmups before collecting one candidate.
        for(int i=0;i<kWarmup;++i)c.launch(1);CUDA_CHECK_KERNEL();
        c.launch(1);CUDA_CHECK_KERNEL();
        std::printf("{\"event\":\"profile_target\",\"shape_id\":\"%dx%dx%d\",\"warmup\":%d}\n",
                    kShapes[id].m,kShapes[id].n,kShapes[id].k,kWarmup);
    } else {
        for(int id=0;id<kShapeCount;++id) {
            Case c(kShapes[id],props.multiProcessorCount,handle);
            if(!c.validate(id,"before"))return 10;
            if(mode=="bench"&&kShapes[id].acceptance)c.timing(id);
            if(!c.validate(id,"after"))return 11;
        }
    }
    BLAS_CHECK(cublasDestroy(handle));
    std::printf("{\"event\":\"complete\",\"mode\":\"%s\"}\n",mode.c_str());
    return 0;
}
