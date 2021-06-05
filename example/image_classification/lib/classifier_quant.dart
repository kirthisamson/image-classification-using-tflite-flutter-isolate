import 'package:imageclassification/classifier.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:tflite_flutter_helper/tflite_flutter_helper.dart';

class ClassifierQuant extends Classifier {
  ClassifierQuant(
      {required Interpreter interpreter,
      int numThreads: 1,
      required List<String> labels})
      : super(interpreter: interpreter, numThreads: numThreads, labels: labels);

  @override
  NormalizeOp get preProcessNormalizeOp => NormalizeOp(0, 1);

  @override
  NormalizeOp get postProcessNormalizeOp => NormalizeOp(0, 255);
}
