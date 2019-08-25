package dl

import java.awt.Color
import java.awt.image.BufferedImage
import java.io.File

import breeze.linalg._
import com.typesafe.scalalogging.Logger
import dl.common.IdxFormatReader
import dl.network.{FullJoinNetwork, FullJoinNetworkConf, NeuralNetwork}
import javax.imageio.ImageIO
import dl.network.NeuralNetwork._

object Instances {

  val logger = Logger(this.getClass)

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
      imagesMatrixInput,
      labelsMatrixInput,
      PracticeConf(10000,0.1,100,NeuralNetwork.Sigmoid,NeuralNetwork.Softmax,NeuralNetwork.CrossEntropyError)
    )
  }

}
