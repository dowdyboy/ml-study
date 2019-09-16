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
    val generateBatMatrixs = (randIndexSeq:Seq[Int],dataMat:DenseMatrix[Double])=>{
      (0 until randIndexSeq.length by batSize).map{i=>
        var seq:scala.collection.mutable.Seq[scala.collection.mutable.Seq[Double]] = scala.collection.mutable.Seq()
        (i until i+batSize).foreach{k=>
          if(k < randIndexSeq.length) {
            var rowSeq:scala.collection.mutable.Seq[Double] = scala.collection.mutable.Seq[Double]()
            dataMat(randIndexSeq(k),::).inner.toScalaVector().foreach(x=>rowSeq = rowSeq :+ x)
            seq = seq :+ rowSeq
          }
        }
        DenseMatrix.tabulate(seq.length,dataMat.cols){(x,y) => seq(x)(y)}
      }
    }

    val updateWeightsOffsets = (batDataMat:DenseMatrix[Double],batLabelMat:DenseMatrix[Double])=>{
      lossLayer.forward(predict(batDataMat),batLabelMat)
      var dout = lossLayer.backward(1.0)
      val reverseLayers = layers.reverse
      (0 until reverseLayers.length).foreach{i => dout = reverseLayers(i).backward(dout)}
      (0 until layers.length).foreach{i=>
        if(layers(i).isInstanceOf[AffineLayer]){
          val affLay = layers(i).asInstanceOf[AffineLayer]
//          logger.info("each update start:")
//          logger.info(s"${affLay.dWeightMat}")
//          logger.info(s"${affLay.dOffsetVec}")
          affLay.weightMat = affLay.weightMat - (affLay.dWeightMat *:* DenseMatrix.tabulate(affLay.dWeightMat.rows,affLay.dWeightMat.cols){(x,y)=>learnRate})
          affLay.offsetVec = affLay.offsetVec - (affLay.dOffsetVec * learnRate)
//          logger.info(s"${affLay.weightMat}")
//          logger.info(s"${affLay.offsetVec}")
//          logger.info("each update end;")
        }
      }
    }

    val randIndexList = new ArrayList[Int]()
    (0 until dataMat.rows).foreach(i => randIndexList.add(i))
    Collections.shuffle(randIndexList)
    val randIndexSeq = randIndexList.asScala

    val batMatrixs = generateBatMatrixs(randIndexSeq,dataMat)
    val labelMatrixs = generateBatMatrixs(randIndexSeq,labelMat)

    (0 until iterNum).foreach{i=>
      val batDataMat = batMatrixs(i % batMatrixs.length)
      val batLabelMat = labelMatrixs(i % batMatrixs.length)
      updateWeightsOffsets(batDataMat,batLabelMat)
    }

//    layers.filter(lay=>lay.isInstanceOf[AffineLayer]).map(_.asInstanceOf[AffineLayer]).foreach(lay=>{
//      println(lay.weightMat)
//      println(lay.offsetVec)
//    })
  }

}
