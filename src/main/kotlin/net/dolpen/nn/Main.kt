package net.dolpen.nn

import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions
import java.nio.file.Path

val graphConfiguration = NeuralNetConfiguration.Builder()
    .seed(42)
    .updater(Adam())
    .l2(0.001)
    .list()
    .layer(
        DenseLayer.Builder()
            .nIn(4).nOut(3)
            .activation(Activation.TANH)
            .weightInit(WeightInit.XAVIER)
            .build()
    )
    .layer(
        OutputLayer.Builder(
            LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD
        )
            .nIn(3).nOut(3)
            .activation(Activation.SOFTMAX)
            .weightInit(WeightInit.XAVIER)
            .build()
    )
    .build()






fun main(){
    val reader = CSVRecordReader(0, '\t')
    reader.initialize(FileSplit(Path.of("data","ifds.tsv").toFile()))
    val iterator = RecordReaderDataSetIterator(reader,150,4,3)

    // GPUを使った場合のシャッフルにバグがあるため使えない
    // val sourceData = iterator.next()
    // sourceData.shuffle(Random(System.currentTimeMillis()).nextLong())
    val seed = iterator.next().asList()
    seed.shuffle()
    val sourceData = DataSet.merge(seed)
    reader.close()

    val normalizer = NormalizerStandardize()
    normalizer.fit(sourceData)
    normalizer.transform(sourceData)
    val splitter = sourceData.splitTestAndTrain(0.65)
    val trainData = ListDataSetIterator(splitter.train.asList())
    val testData = ListDataSetIterator(splitter.test.asList())

    val graph = MultiLayerNetwork(graphConfiguration)
    graph.init()
    graph.fit(trainData, 10)

    val result = graph.evaluate<Evaluation>(testData)
    println(result)




}