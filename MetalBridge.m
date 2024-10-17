//
//  MetalBridge.m
//  fgrain_metal
//
//  Created by yuygfgg on 2024/10/16.
//

// MetalBridge.m
#import "fgrain_metal-Bridging-Header.h"
#import "fgrain_metal-Swift.h"

void runMetalComputationBridge(float* inputData, float* outputData, int width, int height,
                               int stride,
                               int numIterations, float grainRadiusMean, float grainRadiusStd,
                               float sigma, int32_t seed) {
    MetalComputeBridge *bridge = [[MetalComputeBridge alloc] init];
    [bridge runMetalComputationWithInputData:inputData
                                  outputData:outputData
                                       width:width
                                      height:height
                                      stride:stride
                               numIterations:numIterations
                             grainRadiusMean:grainRadiusMean
                              grainRadiusStd:grainRadiusStd
                                       sigma:sigma
                                        seed:seed];
}
