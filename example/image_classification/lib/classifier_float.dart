import 'package:imageclassification/classifier.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:tflite_flutter_helper/tflite_flutter_helper.dart';

class ClassifierFloat extends Classifier {
  ClassifierFloat(
      {required Interpreter interpreter,
      int numThreads: 1,
      required List<String> labels})
      : super(interpreter: interpreter, numThreads: numThreads, labels: labels);

  @override
  NormalizeOp get preProcessNormalizeOp => NormalizeOp(127.5, 127.5);

  @override
  NormalizeOp get postProcessNormalizeOp => NormalizeOp(0, 1);
}
