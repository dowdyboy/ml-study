package dl

import java.awt.Color
import java.awt.image.BufferedImage
import java.io.File

import breeze.linalg._
import dl.common.IdxFormatReader
import dl.network.{FullJoinNetwork, NeuralNetwork}
import javax.imageio.ImageIO
import dl.network.NeuralNetwork._

object Instances {

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

  def testFullJoinNetwork = {
    val imagesIdx = IdxFormatReader.fromFile("assets/mnist/t10k-images.idx3-ubyte")
    val imagesMatrix = DenseMatrix.tabulate(imagesIdx.dimensionsSize(0),imagesIdx.dimensionsSize(1)*imagesIdx.dimensionsSize(2)){(i,j)=>
      imagesIdx.data(i*imagesIdx.dimensionsSize(1)*imagesIdx.dimensionsSize(2)+j).asInstanceOf[Int]
    }
    val labelsIdx = IdxFormatReader.fromFile("assets/mnist/t10k-labels.idx1-ubyte")
    val labelsVector = DenseVector[Int](labelsIdx.data.map(_.toString.toInt).toArray)
    val labelsMatrix = DenseMatrix.tabulate(labelsVector.length,10){(x,y)=>
      if(labelsVector(x) == y) 1 else 0
    }

    val network = new FullJoinNetwork(Seq(784,100,10))
    network.practice(
      imagesMatrix.map(_.toDouble),
      labelsMatrix.map(_.toDouble),
      PracticeConf(1000,0.01,100,NeuralNetwork.Sigmoid,NeuralNetwork.Softmax,NeuralNetwork.CrossEntropyError,Some("args.txt"))
    )
  }

}
