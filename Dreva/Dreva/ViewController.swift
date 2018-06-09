//
//  ViewController.swift
//  Dreva
//
//  Created by Yuan-Pu Hsu on 5/14/18.
//  Copyright © 2018 Yuan-Pu Hsu. All rights reserved.
//

//
//  ViewController.swift
//  Dreva
//
//  Created by Yuan-Pu Hsu on 5/13/18.
//  Copyright © 2018 Yuan-Pu Hsu. All rights reserved.
//

import UIKit
import MapKit
import CoreLocation
import AVFoundation
import CoreML
import Vision
import ImageIO

class ViewController: UIViewController, AVCapturePhotoCaptureDelegate, CLLocationManagerDelegate, UIImagePickerControllerDelegate,  UINavigationControllerDelegate {
    
    @IBOutlet weak var startButton: UIButton!
    @IBOutlet weak var stopButton: UIButton!
    @IBOutlet weak var previewView: UIImageView!
    @IBOutlet weak var currentImagePreviewView: UIImageView!
    @IBOutlet weak var mapView: MKMapView!
    @IBOutlet weak var speedLabel: UILabel!
    @IBOutlet weak var scoreLabel: UILabel!
    @IBOutlet weak var classifier: UILabel!
    
    
    var captureSession = AVCaptureSession()
    var currentCamera: AVCaptureDevice?
    var photoOutput: AVCapturePhotoOutput?
    var cameraPreviewLayer: AVCaptureVideoPreviewLayer?
    var captureTimer: Timer!
    var currentImage: UIImage!
    let clManager = CLLocationManager()
    let imagePickerController = UIImagePickerController()
    var model: traffic_sign_classifier_model!
    var firstRecord: Bool = false

    override func viewDidLoad() {
        super.viewDidLoad()
        self.applyRoundCorner(startButton)
        self.applyRoundCorner(stopButton)
        clManager.delegate = self
        clManager.desiredAccuracy = kCLLocationAccuracyBest
        clManager.requestAlwaysAuthorization()
        clManager.startUpdatingLocation()
        imagePickerController.delegate = self
        model = traffic_sign_classifier_model()
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }
    
    func setupCaptureSession() {
        captureSession.sessionPreset = AVCaptureSession.Preset.photo
    }
    
    func setupDevice() {
        let deviceDiscoverySession = AVCaptureDevice.DiscoverySession(deviceTypes: [AVCaptureDevice.DeviceType.builtInWideAngleCamera], mediaType: AVMediaType.video, position: AVCaptureDevice.Position.unspecified)
        let devices = deviceDiscoverySession.devices
        for device in devices {
            if device.position == AVCaptureDevice.Position.back {
                currentCamera = device
            }
        }
    }
    
    func setupInputOutput() {
        do {
            let captureDeviceInput = try AVCaptureDeviceInput(device: currentCamera!)
            if let inputs = captureSession.inputs as? [AVCaptureDeviceInput] {
                for input in inputs {
                    captureSession.removeInput(input)
                }
            }
            captureSession.addInput(captureDeviceInput)
            photoOutput = AVCapturePhotoOutput()
            photoOutput?.setPreparedPhotoSettingsArray([AVCapturePhotoSettings(format: [AVVideoCodecKey: AVVideoCodecType.jpeg])], completionHandler: nil)
            captureSession.addOutput(photoOutput!)
        } catch {
            print(error)
        }
    }
    
    func setupPreviewLayer() {
        cameraPreviewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        cameraPreviewLayer?.videoGravity = AVLayerVideoGravity.resizeAspectFill
        cameraPreviewLayer?.connection?.videoOrientation = AVCaptureVideoOrientation.portrait
        cameraPreviewLayer?.frame = previewView.bounds
        previewView.layer.addSublayer(cameraPreviewLayer!)
        //        previewView.layer.insertSublayer(cameraPreviewLayer!, at: 0)
    }
    
    func startRunningCaptureSession() {
        captureSession.startRunning()
    }
    
    @objc func captureNow() {
        let settings = AVCapturePhotoSettings()
        let previewPixelType = settings.availablePreviewPhotoPixelFormatTypes.first!
        let previewFormat = [kCVPixelBufferPixelFormatTypeKey as String: previewPixelType, kCVPixelBufferWidthKey as String: 168, kCVPixelBufferHeightKey as String: 168]
        settings.previewPhotoFormat = previewFormat
        photoOutput?.capturePhoto(with: settings, delegate: self)
    }
    
    @IBAction func pressStart(_ sender: Any) {
        if !firstRecord {
            setupCaptureSession()
            setupDevice()
            setupInputOutput()
            setupPreviewLayer()
            startRunningCaptureSession()
            firstRecord = true
        }
        captureTimer = Timer.scheduledTimer(timeInterval: 1, target: self, selector: #selector(captureNow), userInfo: nil, repeats: true)
        self.classifier.text = ""
    }
    @IBAction func pressStop(_ sender: Any) {
        captureTimer.invalidate()
    }
    @IBAction func openLibrary(_ sender: Any) {
        imagePickerController.allowsEditing = true
        imagePickerController.sourceType = .photoLibrary
        present(imagePickerController, animated: true)
    }
    
    @objc func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [String : Any]) {
        let start = DispatchTime.now() // <<<<<<<<<< Start time
        let image = info["UIImagePickerControllerOriginalImage"] as! UIImage
//        UIGraphicsBeginImageContextWithOptions(CGSize(width: 343, height: 343), true, 2.0)
//        image.draw(in: CGRect(x: 0, y: 0, width: 343, height: 343))
//        let newImage = UIGraphicsGetImageFromCurrentImageContext()!
//        libraryPhotoPreview.image = newImage
        UIGraphicsBeginImageContextWithOptions(CGSize(width: 32, height: 32), true, 2.0)
        image.draw(in: CGRect(x: 0, y: 0, width: 32, height: 32))
        let testImage = UIGraphicsGetImageFromCurrentImageContext()!
//        let image = info[UIImagePickerControllerOriginalImage] as! UIImage
//        libraryPhotoPreview.image = image
//        updateClassifications(for: image)
        currentImagePreviewView.image = testImage

        UIGraphicsEndImageContext()

        let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue, kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
        var pixelBuffer : CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, Int(testImage.size.width), Int(testImage.size.height), kCVPixelFormatType_32ARGB, attrs, &pixelBuffer)
        guard (status == kCVReturnSuccess) else {
            return
        }

        CVPixelBufferLockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
        let pixelData = CVPixelBufferGetBaseAddress(pixelBuffer!)

        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(data: pixelData, width: Int(testImage.size.width), height: Int(testImage.size.height), bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer!), space: rgbColorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)

        context?.translateBy(x: 0, y: testImage.size.height)
        context?.scaleBy(x: 1.0, y: -1.0)

        UIGraphicsPushContext(context!)
        testImage.draw(in: CGRect(x: 0, y: 0, width: testImage.size.width, height: testImage.size.height))

        UIGraphicsPopContext()
        CVPixelBufferUnlockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
        imagePickerController.dismiss(animated: true, completion: nil)
        guard let prediction = try? model.prediction(image: pixelBuffer!) else {
            return
        }
        classifier.text = "\(prediction.classLabel)"
        let end = DispatchTime.now()   // <<<<<<<<<<   end time
        let nanoTime = end.uptimeNanoseconds - start.uptimeNanoseconds // <<<<< Difference in nano seconds (UInt64)
        let timeInterval = Double(nanoTime) / 1_000_000_000 // Technically could overflow for long running tests
        print("CNN run time \(timeInterval) seconds")
    }
    
    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        picker.dismiss(animated: true, completion: nil)
    }
    
    func photoOutput(_ captureOutput: AVCapturePhotoOutput,  didFinishProcessingPhoto photoSampleBuffer: CMSampleBuffer?,  previewPhoto previewPhotoSampleBuffer: CMSampleBuffer?, resolvedSettings:  AVCaptureResolvedPhotoSettings, bracketSettings:   AVCaptureBracketedStillImageSettings?, error: Error?) {
        
        if let error = error {
            print("error occure : \(error.localizedDescription)")
        }
        
        if  let sampleBuffer = photoSampleBuffer,
            let previewBuffer = previewPhotoSampleBuffer,
            let dataImage =  AVCapturePhotoOutput.jpegPhotoDataRepresentation(forJPEGSampleBuffer:  sampleBuffer, previewPhotoSampleBuffer: previewBuffer) {
            print(UIImage(data: dataImage)?.size as Any)
            
            let dataProvider = CGDataProvider(data: dataImage as CFData)
            let cgImageRef: CGImage! = CGImage(jpegDataProviderSource: dataProvider!, decode: nil, shouldInterpolate: true, intent: .defaultIntent)
            let image = UIImage(cgImage: cgImageRef, scale: 1.0, orientation: UIImageOrientation.right)
            
            currentImagePreviewView.image = image
        } else {
            print("some error here")
        }
    }
    
    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        let location = locations[0]
        let span = MKCoordinateSpanMake(0.03, 0.03)
        let currentLocation = CLLocationCoordinate2DMake(location.coordinate.latitude, location.coordinate.longitude)
        let region = MKCoordinateRegionMake(currentLocation, span)
        mapView.setRegion(region, animated: true)
        self.mapView.showsUserLocation = true
        speedLabel.text = "\(location.speed)"
    }
    
    func applyRoundCorner(_ object: AnyObject) {
        object.layer.cornerRadius = object.frame.size.width / 2
        object.layer.masksToBounds = true
    }
}



