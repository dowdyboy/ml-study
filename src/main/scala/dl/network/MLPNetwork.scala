package dl.network

import java.util.{ArrayList, Collections}

import breeze.linalg.DenseMatrix
import com.typesafe.scalalogging.Logger

import scala.collection.JavaConverters._
import dl.layer.{AffineLayer, Layer, LossLayer}


class MLPNetwork {

  val logger = Logger(this.getClass)

  private var iterNum:Int = 0
  private var batSize:Int = 0
  private var learnRate:Double = 0.0

  private var layers = Seq[Layer]()
  private var lossLayer:LossLayer = null

  private def generateBatData(dataMat:DenseMatrix[Double],labelMat:DenseMatrix[Double]) = {
    val randIndexList = new ArrayList[Int]()
    (0 until dataMat.rows).foreach(i => randIndexList.add(i))
    Collections.shuffle(randIndexList)
    val randIndexSeq = randIndexList.asScala

    (0 until randIndexSeq.length by batSize).map{i=>
      var dataSeq:scala.collection.mutable.Seq[scala.collection.mutable.Seq[Double]] = scala.collection.mutable.Seq()
      var labelSeq:scala.collection.mutable.Seq[scala.collection.mutable.Seq[Double]] = scala.collection.mutable.Seq()
      (i until i+batSize).foreach{k=>
        if(k < randIndexSeq.length) {
          var dataRowSeq:scala.collection.mutable.Seq[Double] = scala.collection.mutable.Seq[Double]()
          val dataRowVec = dataMat(randIndexSeq(k),::).inner
          (0 until dataRowVec.length).foreach{n=>
            dataRowSeq = dataRowSeq :+ dataRowVec.valueAt(n)
          }
          dataSeq = dataSeq :+ dataRowSeq
          var labelRowSeq:scala.collection.mutable.Seq[Double] = scala.collection.mutable.Seq[Double]()
          val labelRowVec = labelMat(randIndexSeq(k),::).inner
          (0 until labelRowVec.length).foreach{n=>
            labelRowSeq = labelRowSeq :+ labelRowVec.valueAt(n)
          }
          labelSeq = labelSeq :+ labelRowSeq
        }
      }
      (DenseMatrix.tabulate(dataSeq.length,dataMat.cols){(x,y) => dataSeq(x)(y)},
        DenseMatrix.tabulate(labelSeq.length,labelMat.cols){(x,y) => labelSeq(x)(y)})
    }
  }

  private def updateWeightsOffsets(batDataMat:DenseMatrix[Double],batLabelMat:DenseMatrix[Double]) = {
    val loss = lossLayer.forward(predict(batDataMat),batLabelMat)
    var dout = lossLayer.backward(1.0)
    (0 until layers.length).foreach{i => dout = layers(layers.length-1-i).backward(dout)}
    (0 until layers.length).foreach{i=>
      if(layers(i).isInstanceOf[AffineLayer]){
        layers(i).asInstanceOf[AffineLayer].update(learnRate)
      }
    }
  }

  def iterNumber() = iterNum
  def iterNumber(num:Int) = {
    iterNum = num
    this
  }

  def batchSize() = batSize
  def batchSize(size:Int) = {
    batSize = size
    this
  }

  def learningRate() = learnRate
  def learningRate(rate:Double) = {
    learnRate = rate
    this
  }

  def layer(layer:Layer) = {
    layers = layers :+ layer
    this
  }
  def layer(layer:LossLayer) = {
    lossLayer = layer
    this
  }
  def layer() = (layers,lossLayer)

  def predict(dataMat:DenseMatrix[Double]) = {
    var mat:DenseMatrix[Double] = dataMat
    (0 until layers.length).foreach{i=>
      mat = layers(i).forward(mat)
    }
    mat
  }

  def practice(dataMat:DenseMatrix[Double],labelMat:DenseMatrix[Double]) = {
    var batSeq = generateBatData(dataMat,labelMat)
    var batCount = 0
    (0 until iterNum).foreach{i=>
      if(batCount >= batSeq.length){
        batSeq = generateBatData(dataMat,labelMat)
        batCount = 0
      }
      val batMat = batSeq(i % batSeq.length)
      updateWeightsOffsets(batMat._1,batMat._2)
      batCount += 1
    }
  }

}
