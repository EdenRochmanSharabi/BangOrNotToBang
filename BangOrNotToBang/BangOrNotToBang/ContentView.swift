import SwiftUI
import AVFoundation
import Vision
import CoreML

struct ContentView: View {
    @StateObject private var cameraViewModel = CameraViewModel()
    
    var body: some View {
        ZStack {
            CameraPreview(session: cameraViewModel.captureSession)
                .ignoresSafeArea()
            
            // Face detection indicator
            if cameraViewModel.isPersonFaceDetected {
                ZStack {
                    RoundedRectangle(cornerRadius: 20)
                        .stroke(cameraViewModel.isFaceAttractive ? Color.green : Color.red, lineWidth: 10)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .padding(20)
                    
                    // Score display at bottom of screen - commented out as requested
                    /*
                    VStack {
                        Spacer()
                        
                        VStack(spacing: 5) {
                            // Show status text
                            if cameraViewModel.isStabilizingFace {
                                Text("STABILIZING...")
                                    .font(.system(size: 16, weight: .bold))
                                    .foregroundColor(.yellow)
                            } else if cameraViewModel.assessmentCompletedForCurrentFace {
                                Text("LOCKED")
                                    .font(.system(size: 16, weight: .bold))
                                    .foregroundColor(.white)
                            } else {
                                Text("ASSESSING...")
                                    .font(.system(size: 16, weight: .bold))
                                    .foregroundColor(.white)
                            }
                            
                            // Score text
                            Text("Score: \(String(format: "%.2f", cameraViewModel.attractivenessScore))")
                                .font(.system(size: 24, weight: .bold))
                        }
                        .padding(10)
                        .background(Color.black.opacity(0.7))
                        .foregroundColor(cameraViewModel.isFaceAttractive ? .green : .red)
                        .cornerRadius(10)
                        .padding(.bottom, 40)
                    }
                    */
                }
            }
        }
        .onAppear {
            cameraViewModel.checkPermissions()
        }
        .onDisappear {
            cameraViewModel.stopCamera()
        }
    }
}

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession
    
    func makeUIView(context: Context) -> UIView {
        let view = UIView(frame: UIScreen.main.bounds)
        
        let previewLayer = AVCaptureVideoPreviewLayer(session: session)
        previewLayer.frame = view.frame
        previewLayer.videoGravity = .resizeAspectFill
        previewLayer.connection?.videoRotationAngle = 0
        view.layer.addSublayer(previewLayer)
        
        return view
    }
    
    func updateUIView(_ uiView: UIView, context: Context) {}
}

class CameraViewModel: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    @Published var isPersonFaceDetected = false
    @Published var isFaceAttractive = false
    @Published var attractivenessScore: Double = 0.0
    @Published var assessmentCompletedForCurrentFace: Bool = false
    @Published var isStabilizingFace: Bool = false
    
    let captureSession = AVCaptureSession()
    private let videoDataOutput = AVCaptureVideoDataOutput()
    private var faceClassifier: VNCoreMLModel?
    private var lastProcessingTime: Date = Date()
    private let processingInterval: TimeInterval = 0.1 // Process frames at 10fps max
    private let confidenceThreshold: Float = 0.5  // Lowered from 0.65 to 0.5
    
    // For debugging
    private var attractiveCount = 0
    private var unattractiveCcount = 0
    
    // For multithreading and performance
    private let processingQueue = DispatchQueue(label: "com.bangornot.videoProcessing", qos: .userInteractive)
    private let inferenceQueue = DispatchQueue(label: "com.bangornot.inference", qos: .userInitiated)
    
    // For stable face assessment
    private var currentFaceID: String?
    private var faceStabilizationStartTime: Date?
    private var faceStabilizationPeriod: TimeInterval = 0.3 // 300ms to verify stable face
    private var isAssessingFace: Bool = false
    private var assessmentScores: [Double] = []
    private var assessmentStartTime: Date?
    private let assessmentDuration: TimeInterval = 1.0 // 1 second
    
    override init() {
        super.init()
        
        // Debug: List all files in the bundle to check for model
        print("DEBUGGING: Listing all resources in the bundle...")
        let fileManager = FileManager.default
        if let bundlePath = Bundle.main.resourcePath {
            do {
                let contents = try fileManager.contentsOfDirectory(atPath: bundlePath)
                for item in contents.sorted() {
                    print("Found bundle item: \(item)")
                }
            } catch {
                print("Error listing bundle contents: \(error)")
            }
        }
        
        // Load ML model on a background thread to not block UI
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            self?.loadMLModel()
        }
    }
    
    private func loadMLModel() {
        do {
            // Specify compute units to use Neural Engine when available
            let config = MLModelConfiguration()
            config.computeUnits = .all // Use Neural Engine, GPU, and CPU as needed
            
            // First attempt: Try the .mlpackage format (newer models)
            if let modelURL = Bundle.main.url(forResource: "FaceAttractivenessModel", withExtension: "mlpackage") {
                print("Found .mlpackage model at: \(modelURL.path)")
                
                let model = try MLModel(contentsOf: modelURL, configuration: config)
                self.faceClassifier = try VNCoreMLModel(for: model)
                
                print("ML model (.mlpackage) loaded successfully with Neural Engine support")
                return
            }
            
            // Second attempt: Try the .mlmodel format (older models)
            if let modelURL = Bundle.main.url(forResource: "FaceAttractivenessModel", withExtension: "mlmodel") {
                print("Found .mlmodel model at: \(modelURL.path)")
                
                let compiledModelURL = try MLModel.compileModel(at: modelURL)
                let model = try MLModel(contentsOf: compiledModelURL, configuration: config)
                self.faceClassifier = try VNCoreMLModel(for: model)
                
                print("ML model (.mlmodel) loaded successfully with Neural Engine support")
                return
            }
            
            // Third attempt: Try the .mlmodelc format (compiled models)
            if let modelURL = Bundle.main.url(forResource: "FaceAttractivenessModel", withExtension: "mlmodelc") {
                print("Found .mlmodelc compiled model at: \(modelURL.path)")
                
                let model = try MLModel(contentsOf: modelURL, configuration: config)
                self.faceClassifier = try VNCoreMLModel(for: model)
                
                print("ML model (.mlmodelc) loaded successfully with Neural Engine support")
                return
            }
            
            // Fourth attempt: Try with a hardcoded path (just for testing)
            let hardcodedPath = "/Users/edenrochman/Documents/Offline_projects/BangOrNotToBang/BangOrNotToBang/FaceAttractivenessModel.mlmodel"
            let fileManager = FileManager.default
            if fileManager.fileExists(atPath: hardcodedPath) {
                print("Found model at hardcoded path")
                let hardcodedURL = URL(fileURLWithPath: hardcodedPath)
                
                let compiledModelURL = try MLModel.compileModel(at: hardcodedURL)
                let model = try MLModel(contentsOf: compiledModelURL, configuration: config)
                self.faceClassifier = try VNCoreMLModel(for: model)
                
                print("ML model loaded successfully from hardcoded path")
                return
            }
            
            // If we get here, we couldn't find the model
            print("ERROR: Could not find FaceAttractivenessModel in the app bundle or at hardcoded path")
            
        } catch {
            print("Failed to load ML model: \(error)")
        }
    }
    
    func checkPermissions() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            self.setupCamera()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                if granted {
                    DispatchQueue.main.async {
                        self?.setupCamera()
                    }
                }
            }
        default:
            break
        }
    }
    
    func setupCamera() {
        captureSession.beginConfiguration()
        
        // Set capture preset for best balance of performance and quality
        captureSession.sessionPreset = .high
        
        guard let videoDevice = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            return
        }
        
        // Configure camera for high performance
        do {
            try videoDevice.lockForConfiguration()
            if videoDevice.isExposureModeSupported(.continuousAutoExposure) {
                videoDevice.exposureMode = .continuousAutoExposure
            }
            if videoDevice.isAutoFocusRangeRestrictionSupported {
                videoDevice.autoFocusRangeRestriction = .near
            }
            videoDevice.unlockForConfiguration()
        } catch {
            print("Failed to configure camera: \(error)")
        }
        
        do {
            let videoDeviceInput = try AVCaptureDeviceInput(device: videoDevice)
            if captureSession.canAddInput(videoDeviceInput) {
                captureSession.addInput(videoDeviceInput)
            }
        } catch {
            print("Failed to create video input: \(error)")
            return
        }
        
        videoDataOutput.alwaysDiscardsLateVideoFrames = true // Optimize for real-time performance
        videoDataOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)]
        
        if captureSession.canAddOutput(videoDataOutput) {
            captureSession.addOutput(videoDataOutput)
            videoDataOutput.setSampleBufferDelegate(self, queue: processingQueue)
            
            // Set optimal connection settings
            if let connection = videoDataOutput.connection(with: .video) {
                connection.videoRotationAngle = 0 // Portrait orientation (0 degrees)
                connection.isVideoMirrored = false
            }
        }
        
        captureSession.commitConfiguration()
        
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            self?.captureSession.startRunning()
        }
    }
    
    func stopCamera() {
        if captureSession.isRunning {
            captureSession.stopRunning()
        }
    }
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        // Throttle processing to save battery and prevent UI lag
        let currentTime = Date()
        if currentTime.timeIntervalSince(lastProcessingTime) < processingInterval {
            return
        }
        lastProcessingTime = currentTime
        
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        // Process face detection
        processFaceDetection(pixelBuffer)
    }
    
    private func processFaceDetection(_ pixelBuffer: CVPixelBuffer) {
        let faceDetectionRequest = VNDetectFaceRectanglesRequest { [weak self] request, error in
            guard let self = self, error == nil, let results = request.results as? [VNFaceObservation], !results.isEmpty else {
                DispatchQueue.main.async { [weak self] in
                    guard let self = self else { return }
                    self.isPersonFaceDetected = false
                    self.isStabilizingFace = false
                    // Reset face tracking when no face detected
                    self.resetFaceAssessment()
                    self.currentFaceID = nil
                    self.faceStabilizationStartTime = nil
                }
                return
            }
            
            // Person face detected
            DispatchQueue.main.async { [weak self] in
                guard let self = self else { return }
                self.isPersonFaceDetected = true
            }
            
            // Find the best face (largest)
            if let bestFace = results.max(by: { $0.boundingBox.width * $0.boundingBox.height < $1.boundingBox.width * $1.boundingBox.height }) {
                // Generate a face ID based on position
                let faceID = self.generateFaceID(from: bestFace)
                
                // Check if this is a new face
                if self.currentFaceID != faceID {
                    // Debug print when face ID changes
                    print("Face ID changed from \(String(describing: self.currentFaceID)) to \(faceID)")
                    
                    // New face detected, reset assessment
                    self.currentFaceID = faceID
                    self.resetFaceAssessment()
                    
                    // Begin face stabilization period
                    self.faceStabilizationStartTime = Date()
                    
                    // Update UI to show we're stabilizing
                    DispatchQueue.main.async { [weak self] in
                        guard let self = self else { return }
                        self.isStabilizingFace = true
                    }
                } 
                // If we have a stable face ID
                else {
                    // Check if we're still in stabilization period
                    if let stabilizationStart = self.faceStabilizationStartTime {
                        // If we've been stable for the required period, begin assessment
                        if Date().timeIntervalSince(stabilizationStart) >= self.faceStabilizationPeriod {
                            // Face is stable, clear stabilization time
                            self.faceStabilizationStartTime = nil
                            
                            // Update UI to show we're done stabilizing
                            DispatchQueue.main.async { [weak self] in
                                guard let self = self else { return }
                                self.isStabilizingFace = false
                            }
                            
                            // If assessment is not completed, proceed with assessment
                            if !self.assessmentCompletedForCurrentFace {
                                self.checkAttractiveness(pixelBuffer: pixelBuffer, faceObservation: bestFace)
                            }
                        }
                        // Still in stabilization period, don't assess yet
                    }
                    // We already cleared stabilization time, and face is still stable
                    else if !self.assessmentCompletedForCurrentFace {
                        // Continue assessment for stable face
                        self.checkAttractiveness(pixelBuffer: pixelBuffer, faceObservation: bestFace)
                    }
                    // Otherwise we keep showing the locked-in result for completed assessment
                }
            }
        }
        
        // Use option to detect only largest face for performance
        faceDetectionRequest.revision = VNDetectFaceRectanglesRequestRevision3
        
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up, options: [:])
        
        // Run on inference queue to not block camera feed
        inferenceQueue.async {
            try? handler.perform([faceDetectionRequest])
        }
    }
    
    // Generate a unique ID for a face based on its position
    private func generateFaceID(from face: VNFaceObservation) -> String {
        // Divide the screen into just a few regions (2x2 grid) for maximum stability
        // This will make the face ID extremely stable even during significant movement
        
        // Quantize face position to one of a few possible positions for stability
        let grid = 4 // only divide screen into 4 quadrants (2x2)
        let quantizedX = Int(face.boundingBox.midX * CGFloat(grid)) / grid
        let quantizedY = Int(face.boundingBox.midY * CGFloat(grid)) / grid
        
        // Only use 2 size buckets - small and large
        let quantizedSize = face.boundingBox.width > 0.3 ? "large" : "small"
        
        // Create a stable position string
        let facePositionStr = "face-x\(quantizedX)-y\(quantizedY)-\(quantizedSize)"
        print("Face position string: \(facePositionStr)")
        return facePositionStr
    }
    
    // Reset face assessment for a new face
    private func resetFaceAssessment() {
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            self.assessmentCompletedForCurrentFace = false
        }
        
        // Non-UI properties don't need main thread dispatch
        faceStabilizationStartTime = Date() // Reset stabilization period
        isAssessingFace = false
        assessmentScores = []
        assessmentStartTime = nil
    }
    
    // Process face assessment over time
    private func processAssessment(score: Double) {
        // If assessment is already completed for this face, don't update the score
        if assessmentCompletedForCurrentFace {
            return
        }
        
        // Start assessment if not already started
        if !isAssessingFace {
            isAssessingFace = true
            assessmentStartTime = Date()
            assessmentScores = []
            print("Started face assessment")
        }
        
        // Add score to collection
        assessmentScores.append(score)
        
        // Only update UI with interim scores during assessment, not after it's complete
        if !assessmentCompletedForCurrentFace {
            DispatchQueue.main.async { [weak self] in
                guard let self = self else { return }
                self.attractivenessScore = score
                self.isFaceAttractive = score > 3.0
            }
        }
        
        // Check if we've been assessing for the required duration
        if let startTime = assessmentStartTime, Date().timeIntervalSince(startTime) >= assessmentDuration {
            // Skip if we don't have enough samples (which could happen due to processing errors)
            if assessmentScores.isEmpty {
                resetFaceAssessment()
                return
            }
            
            // Calculate average score
            let averageScore = assessmentScores.reduce(0, +) / Double(assessmentScores.count)
            print("Assessment completed with \(assessmentScores.count) samples. Average score: \(String(format: "%.2f", averageScore))")
            
            // Determine final attractiveness
            let threshold = 3.0
            let isFinallyAttractive = averageScore > threshold
            
            // Update UI on main thread with final assessment
            DispatchQueue.main.async { [weak self] in
                guard let self = self else { return }
                self.attractivenessScore = averageScore
                self.isFaceAttractive = isFinallyAttractive
                self.assessmentCompletedForCurrentFace = true
            }
            
            // Reset assessment state but keep the completed flag true
            isAssessingFace = false
        }
    }
    
    private func checkAttractiveness(pixelBuffer: CVPixelBuffer, faceObservation: VNFaceObservation) {
        // If we've already completed assessment for this face, don't do it again
        if assessmentCompletedForCurrentFace {
            return
        }
        
        guard let classifier = self.faceClassifier else {
            print("Face classifier not loaded - skipping attractiveness check")
            return
        }
        
        // Validate the face region is within bounds [0,0,1,1]
        let boundingBox = faceObservation.boundingBox
        if boundingBox.origin.x < 0 || boundingBox.origin.y < 0 || 
           boundingBox.origin.x + boundingBox.width > 1 || 
           boundingBox.origin.y + boundingBox.height > 1 {
            // Face is partially outside the frame, skip processing
            return
        }
        
        let request = VNCoreMLRequest(model: classifier) { [weak self] request, error in
            if let error = error {
                print("Error performing ML request: \(error)")
                return
            }
            
            guard let self = self,
                  let results = request.results else {
                print("No results returned")
                return
            }
            
            // Handle the ML model output
            // For the new regression model, we need to extract the raw score
            if let regressionObservation = results.first as? VNCoreMLFeatureValueObservation {
                // Try different approaches to extract the score
                let featureValue = regressionObservation.featureValue
                
                // 1. Try dictionary approach
                let dictionaryValue = featureValue.dictionaryValue
                if let rawScore = dictionaryValue[""] as? Double {
                    
                    // Apply any post-processing to the score (e.g., clipping to valid range)
                    let processedScore = min(max(rawScore, 1.0), 5.0)  // Clip to range [1.0, 5.0]
                    
                    print("Attractiveness score (dictionary): \(String(format: "%.2f", processedScore))")
                    
                    // Process this score as part of our time-based assessment
                    self.processAssessment(score: processedScore)
                    
                    // Count for debugging
                    let isAttractive = processedScore > 3.0
                    if isAttractive {
                        self.attractiveCount += 1
                    } else {
                        self.unattractiveCcount += 1
                    }
                }
                // Try to extract using multiArrayValue (old version)
                else if let multiArrayValue = featureValue.multiArrayValue,
                        multiArrayValue.count > 0 {
                    
                    let score = multiArrayValue[0].doubleValue
                    
                    // Apply any post-processing to the score if needed
                    let processedScore = min(max(score, 1.0), 5.0)  // Clip to range [1.0, 5.0]
                    
                    print("Attractiveness score (multiarray): \(String(format: "%.2f", processedScore))")
                    
                    // Process this score as part of our time-based assessment
                    self.processAssessment(score: processedScore)
                    
                    // Count for debugging
                    let isAttractive = processedScore > 3.0
                    if isAttractive {
                        self.attractiveCount += 1
                    } else {
                        self.unattractiveCcount += 1
                    }
                } else {
                    print("Could not extract score from feature value")
                    print("Feature value type: \(type(of: featureValue))")
                    print("Feature value: \(featureValue)")
                }
            } else {
                // Fall back to classification approach for backward compatibility
                if let classificationResults = results as? [VNClassificationObservation],
                   !classificationResults.isEmpty {
                    
                    // Process classification results
                    let topResult = classificationResults[0]
                    print("Classification result: \(topResult.identifier) with confidence \(topResult.confidence)")
                    
                    // Determine if attractive based on classification
                    let isAttractive = topResult.identifier == "attractive" && topResult.confidence > self.confidenceThreshold
                    
                    // For classification models, use a binary score (1.0 or 5.0)
                    let binaryScore = isAttractive ? 5.0 : 1.0
                    self.processAssessment(score: binaryScore)
                    
                    // Update UI immediately for classification models if not averaging
                    if !self.isAssessingFace {
                        DispatchQueue.main.async {
                            self.isFaceAttractive = isAttractive
                        }
                    }
                } else {
                    print("Unexpected result type from Core ML model")
                    print("Result type: \(type(of: results.first))")
                    if let firstResult = results.first {
                        print("First result: \(firstResult)")
                    }
                }
            }
        }
        
        // Configure the ML request for best accuracy and performance
        request.regionOfInterest = boundingBox
        request.imageCropAndScaleOption = VNImageCropAndScaleOption.centerCrop
        
        // Update to handle deprecated usesCPUOnly
        if #available(iOS 15.0, *) {
            // Use preferBackgroundProcessing instead of usesCPUOnly
            request.preferBackgroundProcessing = false
        }
        
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up, options: [:])
        inferenceQueue.async {
            do {
                try handler.perform([request])
            } catch {
                print("Error performing vision request: \(error)")
            }
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
} 