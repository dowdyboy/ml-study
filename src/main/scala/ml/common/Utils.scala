package ml.common

import java.io.File

import breeze.linalg._

import scala.io.Source
import scala.util.Random

trait FileSplitType
case object CsvSplitType extends FileSplitType
case object TsvSplitType extends FileSplitType
case class CustomSplitType(split:String) extends FileSplitType


object Utils {

  // 根据文件路径创建数据集
  def createDataSet(path: String,columnCount:Int,labelIndex:Int,fileType:FileSplitType = CsvSplitType) = {
    import java.util.ArrayList
    val source = Source.fromFile(path)
    val dataSetArrayList = new ArrayList[Array[String]]()
    source.getLines().foreach{line=>
      val columnArray = fileType match {
        case CsvSplitType => line.split(",")
        case TsvSplitType => line.split("\t")
        case CustomSplitType(split) => line.split(split)
        case _ => throw new RuntimeException("not support this type")
      }
      if(line != null && !line.equals("") && columnArray.length == columnCount){
        dataSetArrayList.add(columnArray)
      }
    }
    source.close()
    val origDataSet = (dataSetArrayList.size() > 0) match {
      case true => DenseMatrix.tabulate(dataSetArrayList.size(),dataSetArrayList.get(0).length){(i,j)=>(dataSetArrayList.get(i))(j)}
      case false => throw new RuntimeException("there is no data")
    }
    val labelVector = origDataSet(::,labelIndex)
    val targetDataSet = origDataSet.delete(labelIndex,Axis._1)
    (targetDataSet,labelVector,origDataSet)
  }

  // 归一化矩阵
  def norm(dataSet:DenseMatrix[Double]) = {
    // 计算各个特征最小值
    val minVect = min(dataSet,Axis._0).inner
    // 计算各个特征最大值
    val maxVect = max(dataSet,Axis._0).inner
    // 计算各个特征数值范围
    val rangeVect = maxVect - minVect
    // 计算归一化矩阵
    val normDataSet = (dataSet - tile(minVect.t,1,dataSet.rows)) /:/ tile(rangeVect.t,1,dataSet.rows)
    (normDataSet,rangeVect,minVect)
  }

  // 归一化向量
  def norm(vect:DenseVector[Double],minVect:DenseVector[Double],rangeVect:DenseVector[Double]) = {
    (vect - minVect) /:/ rangeVect
  }

  object SimpleBayes {

    def readEmailFile = {
      val hamDir = new File("assets/simple_bayes/ham")
      val spamDir = new File("assets/simple_bayes/spam")
      val hamDataSeq = Random.shuffle(hamDir.listFiles().toSeq).map(f=>{
        Source.fromFile(f,"GBK").getLines().flatMap(line=>{
          line.split("\\W").toSeq.map(_.toLowerCase()).filter(_.length>2)
        }).toSeq
      })
      val hamLabelSeq = Seq.fill(hamDataSeq.length)("ham")
      val spamDataSeq = Random.shuffle(spamDir.listFiles().toSeq).map(f=>{
        Source.fromFile(f,"GBK").getLines().flatMap(line=>{
          line.split("\\W").toSeq.map(_.toLowerCase()).filter(_.length>2)
        }).toSeq
      })
      val spamLabelSeq = Seq.fill(spamDataSeq.length)("spam")
      val trainDataSeq = hamDataSeq.drop(2) ++ spamDataSeq.drop(3)
      val trainLabelSeq = hamLabelSeq.drop(2) ++ spamLabelSeq.drop(3)
      val testDataSeq = hamDataSeq.take(2) ++ spamDataSeq.take(3)
      val testLabelSeq = hamLabelSeq.take(2) ++ spamLabelSeq.take(3)
      (trainDataSeq,trainLabelSeq,testDataSeq,testLabelSeq)
    }
  }

}
