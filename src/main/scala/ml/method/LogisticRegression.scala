package ml.method

import breeze.linalg._
import breeze.numerics.sigmoid

import scala.util.Random

class LogisticRegression(val data:DenseMatrix[Double], val labels:DenseVector[String], alpha:Double = 0.01, numIter:Int = 150) {

  def this(data:Seq[Seq[Double]],labels:Seq[String],alpha:Double,numIter:Int) = {
    this(DenseMatrix.tabulate(data.length,data(0).length){(x,y)=>data(x)(y)},new DenseVector[String](labels.toArray),alpha,numIter)
  }

  private def calcuWeight(data:DenseMatrix[Double], labels:DenseVector[String]) = {
    var weights = DenseVector.ones[Double](data.cols)
    (0 until numIter).foreach{i=>
      var dataIndexList:Seq[Int] = (0 until data.rows)
      (0 until data.rows).foreach{j=>
        val newAlpha = 4.0 / (i+j+1.0) + alpha
        val randomIndex = Random.nextInt(dataIndexList.length)
        val h = sigmoid(sum(data(randomIndex , ::).inner * weights))
        val error = labels(randomIndex).toDouble - h
        weights = weights + newAlpha * error * data(randomIndex,::).inner
        dataIndexList = dataIndexList.filter(_ != dataIndexList(randomIndex))
      }
    }
    weights
  }

  val weights = calcuWeight(data,labels)

  def classify(input:Seq[Double]):Double = classify(DenseVector(input.toArray))

  def classify(input:DenseVector[Double]):Double = {
    if(sigmoid(sum(input * weights)) > 0.5) 1.0
    else 0.0
  }

}
