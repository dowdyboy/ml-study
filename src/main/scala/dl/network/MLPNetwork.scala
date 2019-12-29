package dl.network

import java.util.{ArrayList, Collections}

import breeze.linalg.DenseMatrix

import scala.collection.JavaConverters._
import dl.layer.{AffineLayer, BatchNormLayer, Layer, LossLayer}
import dl.monitor.Monitor
import dl.network.MLPNetwork.{He, InitialWeightValue, Xavier}
import dl.optimizer.Optimizer


object MLPNetwork {
  trait InitialWeightValue
  case object Xavier extends InitialWeightValue
  case object He extends InitialWeightValue
}
class MLPNetwork {

  private var iterNum:Int = 0
  private var batSize:Int = 0
  private var optimizer:Optimizer = null
  private var monitor:Monitor = null

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
    val loss = lossLayer.forward(predict(batDataMat,true),batLabelMat)
    var dout = lossLayer.backward(1.0)
    (0 until layers.length).foreach{i => dout = layers(layers.length-1-i).backward(dout)}
    (0 until layers.length).foreach{i=>
      if(layers(i).isInstanceOf[AffineLayer]){
        val affineLayer = layers(i).asInstanceOf[AffineLayer]
        affineLayer.weightMat = optimizer.update(i,affineLayer.weightMat,affineLayer.dWeightMat)
        affineLayer.offsetVec = optimizer.update(i,affineLayer.offsetVec,affineLayer.dOffsetVec)
      }
      if(layers(i).isInstanceOf[BatchNormLayer]){
        val batchNormLayer = layers(i).asInstanceOf[BatchNormLayer]
        batchNormLayer.gammaMat = optimizer.update(i,batchNormLayer.gammaMat,batchNormLayer.dGammaMat)
        batchNormLayer.betaMat = optimizer.update(i,batchNormLayer.betaMat,batchNormLayer.dBetaMat)
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

  def layer(layer:Layer) = {
    layers = layers :+ layer
    this
  }
  def layer(layer:LossLayer) = {
    lossLayer = layer
    this
  }
  def layer() = (layers,lossLayer)

  def optimize(optimizer: Optimizer) = {
    this.optimizer = optimizer
    this
  }
  def optimize() = optimizer

  def monite(monitor: Monitor) = {
    this.monitor = monitor
    this
  }
  def monite() = monitor

  def weightValues(valueType:InitialWeightValue) = {
    valueType match {
      case Xavier =>
        (0 until layers.length).foreach{i=>
          if(layers(i).isInstanceOf[AffineLayer]){
            val affineLayer = layers(i).asInstanceOf[AffineLayer]
            affineLayer.weightMat = affineLayer.weightMat / Math.sqrt(affineLayer.weightMat.rows.toDouble)
          }
        }
      case He =>
        (0 until layers.length).foreach{i=>
          if(layers(i).isInstanceOf[AffineLayer]){
            val affineLayer = layers(i).asInstanceOf[AffineLayer]
            affineLayer.weightMat = affineLayer.weightMat *:* Math.sqrt(2.0 / affineLayer.weightMat.rows.toDouble)
          }
        }
    }
    this
  }

  def predict(dataMat:DenseMatrix[Double],isTrain:Boolean = false) = {
    var mat:DenseMatrix[Double] = dataMat
    (0 until layers.length).foreach{i=>
      mat = layers(i).forward(mat,isTrain)
    }
    mat
  }

  def practice(dataMat:DenseMatrix[Double],labelMat:DenseMatrix[Double]) = {
    var batSeq = generateBatData(dataMat,labelMat)
    var batCount = 0
    var epochCount = 0
    (0 until iterNum).foreach{i=>
      if(batCount >= batSeq.length){
        batSeq = generateBatData(dataMat,labelMat)
        batCount = 0
        if(monitor != null) monitor.eachEpoch(epochCount)
        epochCount += 1
        System.gc()
      }
      val batMat = batSeq(i % batSeq.length)
      updateWeightsOffsets(batMat._1,batMat._2)
      batCount += 1
      if(monitor != null) monitor.eachIter(i)
    }
  }

}
