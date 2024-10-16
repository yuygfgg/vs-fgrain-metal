//
//  fgrain_metal.cpp
//  fgrain_metal
//
//  Created by yuygfgg on 2024/10/16.
//

#include <stdlib.h>
#include <vector>
#include "VapourSynth4.h"
#include "VSHelper4.h"
#import "fgrain_metal-Bridging-Header.h"
#import "fgrain_metal-Swift.h"

extern "C" {
    void runMetalComputationBridge(float* inputData, float* outputData, int width, int height,
                                   int stride, int numIterations, float grainRadiusMean,
                               float grainRadiusStd, float sigma, int seed);
}

typedef struct {
    VSNode *node;
    const VSVideoInfo *vi;
    int numIterations;
    float grainRadiusMean;
    float grainRadiusStd;
    float sigma;
    int seed;
} FilmGrainData;

static const VSFrame *VS_CC filmGrainGetFrame(int n, int activationReason, void *instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {
    FilmGrainData *d = (FilmGrainData *)instanceData;

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSFrame *src = vsapi->getFrameFilter(n, d->node, frameCtx);
        const VSVideoFormat *fi = vsapi->getVideoFrameFormat(src);
        int height = vsapi->getFrameHeight(src, 0);
        int width = vsapi->getFrameWidth(src, 0);
        VSFrame *dst = vsapi->newVideoFrame(fi, width, height, src, core);

        int plane;
        for (plane = 0; plane < fi->numPlanes; plane++) {
            const uint8_t *srcp = vsapi->getReadPtr(src, plane);
            int srcStride = vsapi->getStride(src, plane);
            uint8_t *dstp = vsapi->getWritePtr(dst, plane);
            int dstStride = vsapi->getStride(dst, plane);

            int h = vsapi->getFrameHeight(src, plane);
            int w = vsapi->getFrameWidth(src, plane);

            if (fi->sampleType == stFloat && fi->bitsPerSample == 32) {
                const float *srcpF32 = (const float *)srcp;
                float *dstpF32 = (float *)dstp;

                runMetalComputationBridge(const_cast<float*>(srcpF32), dstpF32, w, h, srcStride / sizeof(float),
                                          d->numIterations, d->grainRadiusMean, d->grainRadiusStd,
                                          d->sigma, d->seed);
            } else {
                vsapi->freeFrame(src);
                vsapi->freeFrame(dst);
                return NULL;
            }
        }

        vsapi->freeFrame(src);
        return dst;
    }

    return NULL;
}

static void VS_CC filmGrainFree(void *instanceData, VSCore *core, const VSAPI *vsapi) {
    FilmGrainData *d = (FilmGrainData *)instanceData;
    vsapi->freeNode(d->node);
    free(d);
}

static void VS_CC filmGrainCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
    FilmGrainData d;
    FilmGrainData *data;
    int err;

    // 获取输入的视频节点
    d.node = vsapi->mapGetNode(in, "clip", 0, 0);
    d.vi = vsapi->getVideoInfo(d.node);

    // 检查是否支持16-bit int或float
    const VSVideoFormat *fi = &(d.vi->format);
    if (!vsh::isConstantVideoFormat(d.vi) ||
        !(fi->sampleType == stFloat && fi->bitsPerSample == 32)) {
        vsapi->mapSetError(out, "FilmGrain: only 32-bit float input supported");
        vsapi->freeNode(d.node);
        return;
    }

    // 获取并解析参数
    d.numIterations = vsapi->mapGetIntSaturated(in, "numIterations", 0, &err);
    if (err) d.numIterations = 800;

    d.grainRadiusMean = vsapi->mapGetFloat(in, "grainRadiusMean", 0, &err);
    if (err) d.grainRadiusMean = 0.1f;

    d.grainRadiusStd = vsapi->mapGetFloat(in, "grainRadiusStd", 0, &err);
    if (err) d.grainRadiusStd = 0.0f;

    d.sigma = vsapi->mapGetFloat(in, "sigma", 0, &err);
    if (err) d.sigma = 0.8f;

    d.seed = vsapi->mapGetIntSaturated(in, "seed", 0, &err);
    if (err) d.seed = 114514;

    data = (FilmGrainData *)malloc(sizeof(d));
    *data = d;

    VSFilterDependency deps[] = {{d.node, rpStrictSpatial}};
    vsapi->createVideoFilter(out, "FilmGrain", d.vi, filmGrainGetFrame, filmGrainFree, fmParallel, deps, 1, data, core);
}

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin *plugin, const VSPLUGINAPI *vspapi) {
    vspapi->configPlugin("com.yuygfgg.filmgrain", "fgrain_metal", "VapourSynth FilmGrain Plugin", VS_MAKE_VERSION(1, 0), VAPOURSYNTH_API_VERSION, 0, plugin);
    vspapi->registerFunction("Add",
                             "clip:vnode;numIterations:int:opt;grainRadiusMean:float:opt;grainRadiusStd:float:opt;"
                             "sigma:float:opt;seed:int:opt;",
                             "clip:vnode;", filmGrainCreate, NULL, plugin);
}
