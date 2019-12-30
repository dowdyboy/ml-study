package dl

import java.awt.Color
import java.awt.image.BufferedImage
import java.io.File

import breeze.linalg._
import breeze.numerics._
import com.typesafe.scalalogging.Logger
import dl.common.IdxFormatReader
import dl.layer.{AffineLayer, BatchNormLayer, DropoutLayer, ReluLayer, SigmoidLayer, SoftmaxLossLayer}
import dl.monitor.Monitor
import dl.network.MLPNetwork.{He, Xavier}
import dl.network.{FullJoinNetwork, FullJoinNetworkConf, MLPNetwork, NeuralNetwork}
import javax.imageio.ImageIO
import dl.network.NeuralNetwork._
import dl.optimizer.{AdaGradOptimizer, AdamOptimizer, MomentumOptimizer, SGDOptimizer}

object Instances {

  val logger = Logger(this.getClass)

  def getParsedData(trainImagePath:String,trainLabelPath:String) = {
    var imagesIdx = IdxFormatReader.fromFile(trainImagePath)
    var imagesMatrix = DenseMatrix.tabulate(imagesIdx.dimensionsSize(0),imagesIdx.dimensionsSize(1)*imagesIdx.dimensionsSize(2)){(i,j)=>
      imagesIdx.data(i*imagesIdx.dimensionsSize(1)*imagesIdx.dimensionsSize(2)+j).asInstanceOf[Int]
    }
    imagesIdx = null
    var labelsIdx = IdxFormatReader.fromFile(trainLabelPath)
    var labelsVector = DenseVector[Int](labelsIdx.data.map(_.toString.toInt).toArray)
    var labelsMatrix = DenseMatrix.tabulate(labelsVector.length,10){(x,y)=>
      if(labelsVector(x) == y) 1 else 0
    }
    labelsIdx = null

    System.gc()

    val imagesMatrixInput = imagesMatrix.map(_.toDouble)
    imagesMatrix = null
    val labelsMatrixInput = labelsMatrix.map(_.toDouble)
    labelsVector = null
    labelsMatrix = null

    System.gc()

//    val maxVal:Double = max(imagesMatrixInput)
//    (0 until imagesMatrixInput.rows).foreach{i=>
//      (0 until imagesMatrixInput.cols).foreach{k=>
//        imagesMatrixInput.update(i,k,imagesMatrixInput.valueAt(i,k) / maxVal)
//      }
//    }

    (imagesMatrixInput,labelsMatrixInput)
  }

  def idxFormatReaderTest = {
    val idx = IdxFormatReader.fromFile("assets/mnist/t10k-images.idx3-ubyte")
    val mat = DenseMatrix.tabulate(idx.dimensionsSize(0),idx.dimensionsSize(1)*idx.dimensionsSize(2)){(i,j)=>
      idx.data(i*idx.dimensionsSize(1)*idx.dimensionsSize(2)+j).asInstanceOf[Int]
    }
    val imgMat = DenseMatrix.tabulate(28,28){(i,j)=>
      mat(0,::).inner.valueAt(i*28+j)
    }
    val writer = ImageIO.getImageWritersByFormatName("png").next()
    val ios = ImageIO.createImageOutputStream(new File("test.png"))
    val buf_img = new BufferedImage(28,28,BufferedImage.TYPE_INT_RGB)
    val graph = buf_img.getGraphics
    val colorArr = Array.ofDim[Color](28,28)
    (0 until 28).foreach{i=>
      (0 until 28).foreach{j=>
        colorArr(i)(j) = new Color(
          imgMat.valueAt(i,j),
          imgMat.valueAt(i,j),
          imgMat.valueAt(i,j)
        )
        graph.setColor(colorArr(i)(j))
        graph.fillRect(j,i,1,1)
      }
    }
    writer.setOutput(ios)
    writer.write(buf_img)
    (0 until 28).foreach{i=>
      (0 until 28).foreach{j=>
        print((if(imgMat.valueAt(i,j)>0) 1 else 0)+" ")
      }
      println()
    }
  }

  def testFullJoinNetwork(trainImagePath:String,trainLabelPath:String,argsFilePath:String) = {

    val parsedData = getParsedData(trainImagePath,trainLabelPath)

    val network = new FullJoinNetwork(
      Seq(784,100,10),
      FullJoinNetworkConf(
        argsFilePathOpt = Some(argsFilePath),
        onPracticeIterStartOpt = Some(i=>{
          logger.info(s"practice iter ${i} start.")
        }),
        onPracticeIterEndOpt = Some(i=>{
          logger.info(s"practice iter ${i} end.")
        }))
    )
    network.practice(
      parsedData._1,
      parsedData._2,
      PracticeConf(10000,0.1,100,NeuralNetwork.Sigmoid,NeuralNetwork.Softmax,NeuralNetwork.CrossEntropyError)
    )
  }

  def testLayers = {
    logger.info("relu:")
    val reluLayer = new ReluLayer
    reluLayer.forward(DenseMatrix((1.2,-0.2,1.8),(-1.2,0.2,-1.8)),false)
    println(reluLayer.backward(DenseMatrix((9.2,10.8,2.1),(9.2,10.8,2.1))))

    logger.info("sigmoid:")
    val sigmoidLayer = new SigmoidLayer
    sigmoidLayer.forward(DenseMatrix((1.2,-0.2,1.8),(-1.2,0.2,-1.8)),false)
    println(sigmoidLayer.backward(DenseMatrix((9.2,10.8,2.1),(9.2,10.8,2.1))))

    logger.info("affine:")
    val affineLayer = new AffineLayer(
      DenseMatrix((1.0,2.0,3.0),(4.0,5.0,6.0)),
      DenseVector(1.0,2.0,3.0)
    )
    affineLayer.forward(DenseMatrix((1.0,2.0),(3.0,4.0)),false)
    println(affineLayer.backward(DenseMatrix((1.0,2.0,3.0),(4.0,5.0,6.0))))
    println(affineLayer.dWeightMat)
    println(affineLayer.dOffsetVec)

    logger.info("softmax:")
    val softmaxLossLayer = new SoftmaxLossLayer
    val loss = softmaxLossLayer.forward(
      DenseMatrix((0.09,0.9,0.01),(0.8,0.1,0.1),(0.8,0.1,0.1),(0.8,0.1,0.1)),
      DenseMatrix((0.0,0.0,1.0),(0.0,1.0,0.0),(0.0,1.0,0.0),(0.0,1.0,0.0))
    )
    println(loss)
    println(softmaxLossLayer.backward(loss))
  }

  def testMLPNetwork = {

    val trainData = getParsedData("assets/mnist/train-images.idx3-ubyte","assets/mnist/train-labels.idx1-ubyte")
    val parsedData = getParsedData("assets/mnist/t10k-images.idx3-ubyte","assets/mnist/t10k-labels.idx1-ubyte")
    val accurancy = (mlp:MLPNetwork,imageMat:DenseMatrix[Double],labelMat:DenseMatrix[Double])=>{
      var rightCount = 0
      val testResultMat = mlp.predict(imageMat)
      val loss = mlp.layer()._2.forward(testResultMat,labelMat)
      (0 until testResultMat.rows).foreach{i=>
        val testLab = argmax(testResultMat(i,::).inner)
        val rightLab = argmax(labelMat(i,::).inner)
        if(testLab == rightLab) rightCount += 1
      }
      ((rightCount.toDouble / testResultMat.rows.toDouble) * 100,rightCount,loss)
    }

    val mlp = new MLPNetwork
    mlp.iterNumber(10000).batchSize(73).optimize(new AdamOptimizer(0.1))
      .layer(new AffineLayer(DenseMatrix.rand[Double](784,100),DenseVector.zeros[Double](100)))
      .layer(new BatchNormLayer())
      .layer(new ReluLayer)
      .layer(new DropoutLayer())
      .layer(new AffineLayer(DenseMatrix.rand[Double](100,50),DenseVector.zeros[Double](50)))
      .layer(new BatchNormLayer())
      .layer(new ReluLayer)
      .layer(new DropoutLayer())
      .layer(new AffineLayer(DenseMatrix.rand[Double](50,10),DenseVector.zeros[Double](10)))
      //.layer(new ReluLayer)
      .layer(new SoftmaxLossLayer)
      //.weightValues(He)
      .monite(new Monitor {
        override def eachEpoch(epochCount: Int): Unit = {
          val trainRst = accurancy(mlp,trainData._1,trainData._2)
          val testRst = accurancy(mlp,parsedData._1,parsedData._2)
          logger.info(s"epoch ${epochCount} : train data : accurancy(${trainRst._1}%) loss(${trainRst._3})")
          logger.info(s"epoch ${epochCount} : test data : accurancy(${testRst._1}%) loss(${testRst._3})")
        }
      })

    mlp.practice(trainData._1,trainData._2)

  }

}
