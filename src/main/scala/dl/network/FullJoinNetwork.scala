package dl.network

import java.io._
import java.util.{ArrayList, Collections}

import scala.collection.JavaConverters._
import breeze.linalg._
import breeze.numerics._
import dl.network.NeuralNetwork._

class FullJoinNetwork(val layout:Seq[Int]) extends NeuralNetwork {

  var weightsMatrix = scala.collection.mutable.Seq((0 until layout.length-1):_*).map{i=>
    DenseMatrix.rand[Double](layout(i),layout(i+1))
  }
  var offsetsVector = scala.collection.mutable.Seq((0 until layout.length-1):_*).map{i=>
    DenseVector.zeros[Double](layout(i+1))
  }

  private def applyActiveFunction(mat:DenseMatrix[Double],func:NeuralNetwork.ActiveFunction) = {
    func match {
      case NeuralNetwork.Sigmoid => sigmoid(mat)
      case NeuralNetwork.Relu => relu(mat)
      case NeuralNetwork.Identity => mat
      case NeuralNetwork.Softmax => softmax(mat)
      case _ => throw new RuntimeException(s"not support active function : ${func.getClass.getName}")
    }
  }
  private def applyLossFunction(predictResultMat:DenseMatrix[Double],oneHotLabelMat:DenseMatrix[Double],func:NeuralNetwork.LossFunction) = {
    func match {
      case NeuralNetwork.MeanSquaredError => mean_squared_error(predictResultMat,oneHotLabelMat)
      case NeuralNetwork.CrossEntropyError => cross_entropy_error(predictResultMat,oneHotLabelMat)
      case _=> throw new RuntimeException(s"not support loss function : ${func.getClass.getName}")
    }
  }
  private def writeArgs(filePath:String) = {
    val file = new File(filePath)
    file.deleteOnExit()
    file.createNewFile()
    val bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(file)))
    (0 until weightsMatrix.length).foreach{i=>
      bw.write(s"weight:${i}")
      bw.newLine()
      (0 until weightsMatrix(i).rows).foreach{x=>
        val sb = new StringBuilder()
        (0 until weightsMatrix(i).cols).foreach{y=>
          sb.append(s"${weightsMatrix(i).valueAt(x,y)}")
          if(y != weightsMatrix(i).cols-1) sb.append(",")
        }
        bw.write(sb.toString())
        bw.newLine()
      }
      bw.flush()
    }
    (0 until offsetsVector.length).foreach{i=>
      bw.write(s"offset:${i}")
      bw.newLine()
      val sb = new StringBuilder()
      (0 until offsetsVector(i).length).foreach{x=>
        sb.append(s"${offsetsVector(i).valueAt(x)}")
        if(x != offsetsVector(i).length-1) sb.append(",")
      }
      bw.write(sb.toString())
      bw.newLine()
      bw.flush()
    }
    bw.close()
  }

  def predict(mat:DenseMatrix[Double],conf:NeuralNetwork.PredictConf):DenseMatrix[Double] = {
    var inMat = mat
    (0 until weightsMatrix.length).foreach{i=>
      inMat = inMat * weightsMatrix(i)
      inMat = inMat + tile(offsetsVector(i).t,1,inMat.rows)
      inMat = applyActiveFunction(inMat,if(i==weightsMatrix.length-1) conf.outputFunction else conf.activeFunction)
    }
    inMat
  }
  def predict(vec:DenseVector[Double],conf:NeuralNetwork.PredictConf):DenseMatrix[Double] = {
    predict(vec.toDenseMatrix,conf)
  }
  def predict(in:SeqWrapper[Double], conf:NeuralNetwork.PredictConf):DenseMatrix[Double] = {
    in match {
      case SeqVector(seq) => predict(DenseVector[Double](seq.toArray),conf)
      case SeqMatrix(seq) => predict(DenseMatrix.tabulate(seq.length,seq(0).length){(row,col)=>seq(row)(col)},conf)
      case _ => throw new RuntimeException(s"not support this type of SeqWrapper")
    }
  }

  def practice(dataMat:DenseMatrix[Double],labelMat:DenseMatrix[Double],conf:NeuralNetwork.PracticeConf):Unit = {
    val generateBatMatrixs = (randIndexSeq:Seq[Int],dataMat:DenseMatrix[Double])=>{
      (0 until randIndexSeq.length by conf.batSize).map{i=>
        var seq:scala.collection.mutable.Seq[scala.collection.mutable.Seq[Double]] = scala.collection.mutable.Seq()
        (i until i+conf.batSize).foreach{k=>
          var rowSeq:scala.collection.mutable.Seq[Double] = scala.collection.mutable.Seq[Double]()
          if(k < randIndexSeq.length) {
            dataMat(randIndexSeq(k),::).inner.toScalaVector().foreach(x=>rowSeq = rowSeq :+ x)
            seq = seq :+ rowSeq
          }
        }
        DenseMatrix.tabulate(seq.length,dataMat.cols){(x,y) => seq(x)(y)}
      }
    }

    val numericalWeightGrad = (func:()=>Double) => {
      val h:Double = 0.0001
      (0 until weightsMatrix.length).map{x=>
        val gradMat = DenseMatrix.zeros[Double](weightsMatrix(x).rows,weightsMatrix(x).cols)
        (0 until gradMat.rows).foreach{i=>
          (0 until gradMat.cols).foreach{k=>
            val oldVal = weightsMatrix(x).valueAt(i,k)
            weightsMatrix(x).update(i,k,oldVal+h)
            val fn1 = func()
            weightsMatrix(x).update(i,k,oldVal-h)
            val fn2 = func()
            gradMat.update(i,k,(fn1 - fn2) / (2 * h))
            weightsMatrix(x).update(i,k,oldVal)
          }
        }
        gradMat
      }
    }
    val numericalOffsetGrad = (func:()=>Double) => {
      val h:Double = 0.0001
      (0 until offsetsVector.length).map{x=>
        val gradVec = DenseVector.zeros[Double](offsetsVector(x).length)
        (0 until gradVec.length).foreach{i=>
          val oldVal = offsetsVector(x).valueAt(i)
          offsetsVector(x).update(i,oldVal + h)
          val fn1 = func()
          offsetsVector(x).update(i,oldVal - h)
          val fn2 = func()
          //println(s"${i} : ${oldVal},${fn1},${fn2} : ${(fn1 - fn2) / (2 * h)}")
          gradVec.update(i,(fn1 - fn2) / (2 * h))
          offsetsVector(x).update(i,oldVal)
        }
        gradVec
      }
    }

    val generateLossFunc = (batDataMat:DenseMatrix[Double],batLabelMat:DenseMatrix[Double]) => {
      ()=>{
        val predictMat = predict(
          batDataMat,
          NeuralNetwork.PredictConf(conf.activeFunction,conf.outputFunction)
        )
        conf.lossFunction match {
          case NeuralNetwork.MeanSquaredError => sum(mean_squared_error(predictMat,batLabelMat)) / predictMat.rows.toDouble
          case NeuralNetwork.CrossEntropyError => sum(cross_entropy_error(predictMat,batLabelMat)) / predictMat.rows.toDouble
          case _ => throw new RuntimeException(s"not support this loss function : ${conf.lossFunction.getClass.getName}")
        }
      }
    }

    val randIndexList = new ArrayList[Int]()
    (0 until dataMat.rows).foreach(i => randIndexList.add(i))
    Collections.shuffle(randIndexList)
    val randIndexSeq = randIndexList.asScala

    val batMatrixs = generateBatMatrixs(randIndexSeq,dataMat)
    val labelMatrixs = generateBatMatrixs(randIndexSeq,labelMat)

    (0 until conf.iterNumber).foreach{i=>
      println(s"iter ${i} start.")
      val batDataMat = batMatrixs(i % batMatrixs.length)
      val batLabelMat = labelMatrixs(i % batMatrixs.length)
      val lossFunc = generateLossFunc(batDataMat,batLabelMat)
      val weightsGradMatrix = numericalWeightGrad(lossFunc)
      val offsetsGradVector = numericalOffsetGrad(lossFunc)
      (0 until weightsMatrix.length).foreach{i =>
        weightsMatrix.update(i,weightsMatrix(i) - (weightsGradMatrix(i) *:* DenseMatrix.tabulate(weightsGradMatrix(i).rows,weightsGradMatrix(i).cols){(x,y)=> conf.learnRate}))
      }
      (0 until offsetsVector.length).foreach{i =>
        offsetsVector.update(i,offsetsVector(i) - (offsetsGradVector(i) * conf.learnRate))
      }
      if(conf.argsFileNameOpt != None) writeArgs(conf.argsFileNameOpt.get)
      println(s"iter ${i} finished!")
    }
  }
  def practice(data:Seq[Seq[Double]],labels:Seq[Seq[Double]],conf:NeuralNetwork.PracticeConf):Unit = {
    practice(
      DenseMatrix.tabulate(data.length,data(0).length){(x,y) => data(x)(y)},
      DenseMatrix.tabulate(labels.length,labels(0).length){(x,y) => labels(x)(y)},
      conf
    )
  }

  def readArgs(filePath:String) = {
    var curType = ""
    var curIdx = -1
    var curRow = 0
    val file = new File(filePath)
    val br = new BufferedReader(new InputStreamReader(new FileInputStream(file)))
    var line = br.readLine()
    while (line!=null) {
      if(line.startsWith("weight:") || line.startsWith("offset:")){
        curType = (line.split(":"))(0)
        curIdx = (line.split(":"))(1).toInt
        curRow = 0
      }else{
        val dataArr = line.split(",")
        if(curType == "weight"){
          (0 until weightsMatrix(curIdx).cols).foreach{y =>
            weightsMatrix(curIdx).update(curRow,y,dataArr(y).toDouble)
          }
          curRow += 1
        }else if(curType == "offset"){
          (0 until offsetsVector(curIdx).length).foreach{i =>
            offsetsVector(curIdx).update(i,dataArr(i).toDouble)
          }
        }
      }
      line = br.readLine()
    }
    br.close()
  }

}
