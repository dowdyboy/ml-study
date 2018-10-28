package ml

import breeze.linalg.DenseVector
import ml.common.Utils
import ml.method.Knn

object Main {

  def main(args: Array[String]): Unit = {
    val (dataSet,label,_) = Utils.createDataSet("assets/knn/sample.txt",2)
    val knn = new Knn(dataSet.map(_.toDouble),label,3)
    knn.classify(DenseVector[Double](0.6,0.5))
  }

}
