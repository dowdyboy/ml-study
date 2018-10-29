package ml

import ml.common.{TsvSplitType, Utils}
import ml.method.Knn

object Main {

  // knn算法分类约会信息
  def knnDatingClassify() = {
    val (dataSet,label,_) = Utils.createDataSet("assets/knn/datingTrainSet.txt",4,3,TsvSplitType)
    val knn = new Knn(dataSet.map(_.toDouble),label,10)
    val (testDataSet,testLabel,_) = Utils.createDataSet("assets/knn/datingTestSet.txt",4,3,TsvSplitType)
    var rightCount = 0
    for(i <- 0 to (testDataSet.rows-1)){
      val classifyLabel = knn.classify(testDataSet(i,::).inner.map(_.toDouble))
      val realLabel = testLabel(i)
      if(classifyLabel == realLabel) rightCount += 1
      println(s"classify label is ${classifyLabel} , real label is ${realLabel}")
    }
    println(s"the right percent is ${rightCount.toDouble / testDataSet.rows.toDouble * 100}%")
  }

  // knn算法分类约会信息（使用归一化特征量）
  def knnDatingClassifyWithNorm() = {
    val (dataSet,label,_) = Utils.createDataSet("assets/knn/datingTrainSet.txt",4,3,TsvSplitType)
    val (normDataSet,normRangeVect,normMinVect) = Utils.norm(dataSet.map(_.toDouble))
    val knn = new Knn(normDataSet,label,10)
    val (testDataSet,testLabel,_) = Utils.createDataSet("assets/knn/datingTestSet.txt",4,3,TsvSplitType)
    var rightCount = 0
    for(i <- 0 to (testDataSet.rows-1)){
      val classifyLabel = knn.classify(Utils.norm(testDataSet(i,::).inner.map(_.toDouble),normMinVect,normRangeVect))
      val realLabel = testLabel(i)
      if(classifyLabel == realLabel) rightCount += 1
      println(s"classify label is ${classifyLabel} , real label is ${realLabel}")
    }
    println(s"the right percent is ${rightCount.toDouble / testDataSet.rows.toDouble * 100}%")
  }

  def main(args: Array[String]): Unit = {

  }

}
