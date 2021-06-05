import 'dart:io';
import 'dart:isolate';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'package:imageclassification/classifier.dart';
import 'package:imageclassification/classifier_float.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:tflite_flutter_helper/tflite_flutter_helper.dart';

/// Manages separate Isolate instance for inference
class IsolateUtils {
  static const String DEBUG_NAME = "InferenceIsolate";

  late Isolate _isolate;
  ReceivePort _mainIsolateReceivePort = ReceivePort();
  late SendPort _sendPort;

  SendPort? get sendPort => _sendPort;

  Future<void> start() async {
    _isolate = await Isolate.spawn<SendPort>(
      entryPoint,
      _mainIsolateReceivePort.sendPort,
      debugName: DEBUG_NAME,
    );

    _sendPort = await _mainIsolateReceivePort.first;
  }

  static void entryPoint(SendPort mainIsolateSendPort) async {
    final childIsolateReceivePort = ReceivePort();
    mainIsolateSendPort.send(childIsolateReceivePort.sendPort);

    await for (final IsolateData isolateData in childIsolateReceivePort) {
      if (isolateData != null) {
        print("Creating classifier in isolate");
        ClassifierFloat classifier = ClassifierFloat(
            interpreter:
                Interpreter.fromAddress(isolateData.interpreterAddress),
            numThreads: 1,
            labels: isolateData.labels);
        print("Classifier created successfully in isolate. Predicting...");
        Category category = classifier.predict(isolateData.tensorImage);
        // Uncomment this to simulate a delay in the isolate processing time
        // print(
        //     "==> Delaying for 20 seconds to see if it will work in the background");
        // await Future.delayed(Duration(seconds: 20));
        // print("Sending the resulting classification back to the main isolate");
        isolateData.parentRecieverSendPort.send(category);
      }
    }
  }
}

/// Bundles data to pass between Isolate
class IsolateData {
  img.Image tensorImage;
  int interpreterAddress;
  List<String> labels;
  SendPort parentRecieverSendPort;

  IsolateData(this.tensorImage, this.interpreterAddress, this.labels,
      this.parentRecieverSendPort);
}
