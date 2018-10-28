package ml.method

import breeze.linalg._

class Knn(
                protected val dataSet:DenseMatrix[Double],
                protected val labels:DenseVector[String],
                protected val k:Int = 1) {

  val dataSetSize = dataSet.rows
  val dataSetDimension = dataSet.cols

  // 分类器函数
  def classify(in:DenseVector[Double]) = {
    // 计算距离向量
    val distanceVector = sum((tile(in.t,1,dataSetSize) - dataSet).map(d=>d*d),Axis._1).map(d => Math.sqrt(d))
    // argtopk函数的排序，此处有坑
    implicit val ordering = Ordering[Double].reverse
    val countMap = scala.collection.mutable.Map[String,Int]()
    // 计算各个标签出现的次数
    argtopk(distanceVector,k).foreach{idx =>
      countMap.update(labels(idx),countMap.getOrElse(labels(idx),0)+1)
    }
    // 返回出现次数最多的标签
    countMap.toList.sortBy(_._2)(Ordering[Int].reverse).head._1
  }

}
