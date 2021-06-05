import 'dart:io';
import 'dart:isolate';
import 'package:image/image.dart' as img;
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:imageclassification/classifier.dart';
import 'package:imageclassification/classifier_float.dart';
import 'package:logger/logger.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:tflite_flutter_helper/tflite_flutter_helper.dart';

import 'Utils/isolateUtil.dart';

void main() => runApp(MyApp());

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Image Classification',
      theme: ThemeData(
        primarySwatch: Colors.orange,
      ),
      home: MyHomePage(title: 'Flutter Demo Home Page'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  MyHomePage({Key? key, this.title}) : super(key: key);

  final String? title;

  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  String get modelName => 'tup_model_float.tflite';
  final String _labelsFileName = 'assets/tup_labels.txt';

  var logger = Logger();

  File? _image;
  final picker = ImagePicker();

  Image? _imageWidget;

  img.Image? fox;

  Category? category;

  List<String> labels = [];

  late Classifier classifier;

  @override
  void initState() {
    super.initState();
    createClassifier(numThreads: 1);
  }

  Future getImage() async {
    final pickedFile = await picker.getImage(source: ImageSource.gallery);

    _predict();
    setState(() {
      _image = File(pickedFile!.path);
      _imageWidget = Image.file(_image!);
    });
  }

  Future<void> _loadLabels() async {
    labels = await FileUtil.loadLabels(_labelsFileName);
    print(labels);
    if (labels.length > 0) {
      print(labels);
      print('Labels loaded successfully');
    } else {
      print('Unable to load labels');
    }
  }

  void _predict() async {
    await _loadLabels();
    img.Image imageInput = img.decodeImage(_image!.readAsBytesSync())!;
    var pred = await inference(imageInput, classifier);
    setState(() {
      this.category = pred;
    });
  }

  createClassifier({int numThreads = 1}) async {
    await _loadLabels();
    InterpreterOptions _interpreterOptions = InterpreterOptions()
      ..threads = numThreads;
    Interpreter interpreter =
        await Interpreter.fromAsset(modelName, options: _interpreterOptions);
    print('Interpreter Created Successfully');
    classifier = ClassifierFloat(interpreter: interpreter, labels: labels);
  }

  /// Runs inference in another isolate
  Future<Category> inference(
      img.Image imageInput, Classifier classifier) async {
    // Create a ReceivePort from the parent's side
    ReceivePort parentReceivePort = ReceivePort();
    // Build IsolateData object
    var isolateData = IsolateData(imageInput, classifier.interpreter.address,
        classifier.labels, parentReceivePort.sendPort);

    // instantiate IsolateUtils
    IsolateUtils isolateUtils = IsolateUtils();
    await isolateUtils.start();

    // Send IsolateData to the isolate
    isolateUtils.sendPort!
        .send(isolateData..parentRecieverSendPort = parentReceivePort.sendPort);

    // Wait on our recieveport for the result once they are done
    Category category = await parentReceivePort.first;
    return category;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('TfLite Flutter Helper',
            style: TextStyle(color: Colors.white)),
      ),
      body: Column(
        children: <Widget>[
          Center(
            child: _image == null
                ? Text('No image selected.')
                : Container(
                    constraints: BoxConstraints(
                        maxHeight: MediaQuery.of(context).size.height / 2),
                    decoration: BoxDecoration(
                      border: Border.all(),
                    ),
                    child: _imageWidget,
                  ),
          ),
          SizedBox(
            height: 36,
          ),
          Text(
            category != null ? category!.label : '',
            style: TextStyle(fontSize: 20, fontWeight: FontWeight.w600),
          ),
          SizedBox(
            height: 8,
          ),
          Text(
            category != null
                ? 'Confidence: ${category!.score.toStringAsFixed(3)}'
                : '',
            style: TextStyle(fontSize: 16),
          ),
        ],
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: getImage,
        tooltip: 'Pick Image',
        child: Icon(Icons.add_a_photo),
      ),
    );
  }
}
