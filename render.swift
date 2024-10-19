//
//  render.swift
//  fgrain_metal
//
//  Created by yuygfgg on 2024/10/16.
//

import Cocoa
import Metal
import MetalKit
import CoreImage
import simd

@objc class MetalComputeBridge: NSObject {
    
    // Helper function to generate Gaussian random numbers using Box-Muller transform
    func generateGaussianRandom(mean: Float, stdDev: Float) -> Float {
        let u1 = Float.random(in: 0.0...1.0)
        let u2 = Float.random(in: 0.0...1.0)
        
        let z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * Float.pi * u2)
        // Adjust for mean and standard deviation
        return z0 * stdDev + mean
    }
    
    // Helper function to print pixel values from a texture
    func printTexturePixelValues(texture: MTLTexture) {
        let width = texture.width
        let height = texture.height

        // Check the pixel format to handle different texture formats
        if texture.pixelFormat == .r32Float {
            // Single channel (grayscale) texture
            let pixelByteCount = MemoryLayout<Float>.size

            // Create a buffer to hold the pixel data
            var pixelData = [Float](repeating: 0.0, count: width * height)

            // Get the bytes from the texture into the buffer
            texture.getBytes(&pixelData,
                             bytesPerRow: width * pixelByteCount,
                             from: MTLRegionMake2D(0, 0, width, height),
                             mipmapLevel: 0)

            // Print the grayscale pixel values
            for y in 0..<height {
                for x in 0..<width {
                    let index = y * width + x
                    let gray = pixelData[index]
                    if(gray != 0){
                        print("Pixel at (\(x), \(y)): Grayscale=\(gray)")
                    }
                }
            }
        } else {
            print("Unsupported pixel format: \(texture.pixelFormat)")
        }
    }
    
    // Helper function to create a Metal texture from raw grayscale data (single channel)
    func createTextureFromGrayscaleData(device: MTLDevice, data: UnsafePointer<Float>, width: Int, height: Int) -> MTLTexture? {
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float, width: width, height: height, mipmapped: false)
        textureDescriptor.usage = [.shaderRead, .shaderWrite]
        
        guard let texture = device.makeTexture(descriptor: textureDescriptor) else {
            print("Failed to create texture.")
            return nil
        }
        
        let region = MTLRegionMake2D(0, 0, width, height)
        texture.replace(region: region, mipmapLevel: 0, withBytes: data, bytesPerRow: width * MemoryLayout<Float>.size)
        
        return texture
    }
    
    // Function to export Metal texture data to a float array (grayscale)
    func exportTextureToGrayscaleData(texture: MTLTexture) -> [Float]? {
//        printTexturePixelValues(texture: texture)
        let width = texture.width
        let height = texture.height
        let pixelCount = width * height
        var pixelData = [Float](repeating: 0.0, count: pixelCount)
        
        let region = MTLRegionMake2D(0, 0, width, height)
        texture.getBytes(&pixelData, bytesPerRow: width * MemoryLayout<Float>.size, from: region, mipmapLevel: 0)
        
        return pixelData
    }
    
    // Ensure this method is exposed to Objective-C++ by adding @objc
    @objc func runMetalComputation(inputData: UnsafePointer<Float>,
                                   outputData: UnsafeMutablePointer<Float>,
                                   width: Int32,
                                   height: Int32,
                                   stride: Int32,
                                   numIterations: Int32,
                                   grainRadiusMean: Float,
                                   grainRadiusStd: Float,
                                   sigma: Float,
                                   seed: Int32) {
        guard let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue() else {
            print("Failed to create Metal device or command queue.")
            return
        }

//        let dylibPath = URL(fileURLWithPath: #file).deletingLastPathComponent()

        let metallibURL = URL(fileURLWithPath: "/usr/local/lib/vapoursynth/default.metallib")

        // Create Metal device
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("Failed to create Metal device.")
            return
        }

        guard let library = try? device.makeLibrary(URL: metallibURL) else {
            print("Failed to load Metal library at \(metallibURL.path)")
            return
        }

        guard let function = library.makeFunction(name: "film_grain_rendering_kernel") else {
            print("Failed to load Metal kernel function.")
            return
        }

        let pipelineState: MTLComputePipelineState
        do {
            pipelineState = try device.makeComputePipelineState(function: function)
        } catch {
            print("Failed to create pipeline state: \(error)")
            return
        }

        // Create input and output textures from grayscale data
        guard let inputTexture = createTextureFromGrayscaleData(device: device, data: inputData, width: Int(width), height: Int(height)) else {
            print("Failed to create inputTextures.")
            return
        }
        
        // Create an output texture using the input texture's dimensions and format.
        guard let outputTexture = device.makeTexture(descriptor: MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: inputTexture.pixelFormat,  // Use the input texture's pixel format
            width: inputTexture.width,              // Use the input texture's width
            height: inputTexture.height,            // Use the input texture's height
            mipmapped: false)) else {
            print("Failed to create output texture.")
            return
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            print("Failed to create command buffer or encoder.")
            return
        }

        computeEncoder.setComputePipelineState(pipelineState)
        computeEncoder.setTexture(inputTexture, index: 0)
        computeEncoder.setTexture(outputTexture, index: 1)

        var stride = stride
        // Compute ag
        let ag: Float = 1.0 / ceil(1.0 / grainRadiusMean)

        // Initialize arrays for lambda and exp_lambda
        var lambda = [Float](repeating: 0.0, count: 256)
        var exp_lambda = [Float](repeating: 0.0, count: 256)

        // Calculate lambda and exp_lambda
        for i in 0..<256 {
            // Compute lambda values
            lambda[i] = -((ag * ag) / (
                Float.pi * (grainRadiusMean * grainRadiusMean + grainRadiusStd * grainRadiusStd)
            )) * log((255.0 - Float(i)) / 255.1)
            // Compute exp_lambda values
            exp_lambda[i] = exp(-lambda[i])
        }
        
        var x_gaussian = [Float](repeating: 0.0, count: Int(numIterations))
        var y_gaussian = [Float](repeating: 0.0, count: Int(numIterations))
        
        // Initialize random Gaussian offsets
        for i in 0..<Int(numIterations) {
            x_gaussian[i] = generateGaussianRandom(mean: 0.0, stdDev: sigma)
            y_gaussian[i] = generateGaussianRandom(mean: 0.0, stdDev: sigma)
        }
        
        // Create buffers for parameters
        guard let lambdaBuffer = device.makeBuffer(bytes: lambda, length: lambda.count * MemoryLayout<Float>.size, options: []),
              let expLambdaBuffer = device.makeBuffer(bytes: exp_lambda, length: exp_lambda.count * MemoryLayout<Float>.size, options: []),
              let xGaussianBuffer = device.makeBuffer(bytes: x_gaussian, length: x_gaussian.count * MemoryLayout<Float>.size, options: []),
              let yGaussianBuffer = device.makeBuffer(bytes: y_gaussian, length: y_gaussian.count * MemoryLayout<Float>.size, options: []) else {
            print("Failed to create parameter buffers.")
            return
        }
        
        var width = width
        var height = height
        var numIterations = numIterations
        var grainRadiusMean = grainRadiusMean
        var grainRadiusStd = grainRadiusStd
        var sigma = sigma
        var seed = seed

        computeEncoder.setBytes(&width, length: MemoryLayout<Int32>.size, index: 0)
        computeEncoder.setBytes(&height, length: MemoryLayout<Int32>.size, index: 1)
        computeEncoder.setBytes(&stride, length: MemoryLayout<Int32>.size, index: 2)
        computeEncoder.setBytes(&numIterations, length: MemoryLayout<Int32>.size, index: 3)
        computeEncoder.setBytes(&grainRadiusMean, length: MemoryLayout<Float>.size, index: 4)
        computeEncoder.setBytes(&grainRadiusStd, length: MemoryLayout<Float>.size, index: 5)
        computeEncoder.setBytes(&sigma, length: MemoryLayout<Float>.size, index: 6)
        computeEncoder.setBytes(&seed, length: MemoryLayout<Int32>.size, index: 7)
        computeEncoder.setBuffer(lambdaBuffer, offset: 0, index: 8)
        computeEncoder.setBuffer(expLambdaBuffer, offset: 0, index: 9)
        computeEncoder.setBuffer(xGaussianBuffer, offset: 0, index: 10)
        computeEncoder.setBuffer(yGaussianBuffer, offset: 0, index: 11)

        let threadGroupSize = MTLSize(width: 24, height: 24, depth: 1)
        let threadGroups = MTLSize(width: (Int(width) + threadGroupSize.width - 1) / threadGroupSize.width,
                                   height: (Int(height) + threadGroupSize.height - 1) / threadGroupSize.height,
                                   depth: 1)

        computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        computeEncoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        if let outputPixelData = exportTextureToGrayscaleData(texture: outputTexture) {
            let count = outputPixelData.count
            
            outputPixelData.withUnsafeBufferPointer { pixelDataBuffer in
                guard let pixelDataBaseAddress = pixelDataBuffer.baseAddress else { return }
                
                memcpy(outputData, pixelDataBaseAddress, count * MemoryLayout<Float32>.size)
            }
        } else {
            print("Failed to export output texture data.")
        }
    }
}
