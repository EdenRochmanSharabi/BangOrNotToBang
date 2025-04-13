import Foundation
import CoreML

// This class helps Xcode recognize the model and include it in the bundle
class ModelCheck {
    static func ensureModelIsLoaded() {
        let modelURL = Bundle.main.url(forResource: "FaceAttractivenessModel", withExtension: "mlmodel")
        print("Model URL: \(String(describing: modelURL))")
    }
} 