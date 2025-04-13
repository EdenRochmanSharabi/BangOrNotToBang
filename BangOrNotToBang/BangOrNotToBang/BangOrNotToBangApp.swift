import SwiftUI

@main
struct BangOrNotToBangApp: App {
    init() {
        // Check if the ML model file exists in the bundle
        checkModelFile()
    }
    
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
    
    private func checkModelFile() {
        // Check for compiled model first (this is what's actually used at runtime)
        let compiledModelURL = Bundle.main.url(forResource: "FaceAttractivenessModel", withExtension: "mlmodelc")
        if let modelURL = compiledModelURL {
            print("Compiled model file exists at: \(modelURL.path)")
            return
        }
        
        // Fall back to checking for source model
        let sourceModelURL = Bundle.main.url(forResource: "FaceAttractivenessModel", withExtension: "mlmodel")
        if let modelURL = sourceModelURL {
            print("Source model file exists at: \(modelURL.path)")
            return
        }
        
        // If we get here, no model was found
        print("WARNING: Model file not found in bundle!")
        
        // List all resources in the bundle to debug
        if let resourcePath = Bundle.main.resourcePath {
            do {
                let fileManager = FileManager.default
                let items = try fileManager.contentsOfDirectory(atPath: resourcePath)
                print("Bundle contains \(items.count) items:")
                for item in items.sorted() {
                    print("  - \(item)")
                }
            } catch {
                print("Error listing bundle contents: \(error)")
            }
        }
    }
} 