package ml.common

import breeze.linalg.{Axis, DenseMatrix}

import scala.io.Source

trait FileSplitType
case object CsvSplitType extends FileSplitType
case object TsvSplitType extends FileSplitType
case class CustomSplitType(split:String) extends FileSplitType


object Utils {

  // 根据文件路径创建数据集
  def createDataSet(path: String,labelIndex:Int,fileType:FileSplitType = CsvSplitType) = {
    val source = Source.fromFile(path)
    val dataSetArray:Array[Array[String]] = source.getLines().map{line =>
      fileType match {
        case CsvSplitType => line.split(",")
        case TsvSplitType => line.split("\t")
        case CustomSplitType(split) => line.split(split)
        case _ => throw new RuntimeException("not support this type")
      }
    }.toArray
    source.close()
    val origDataSet = (dataSetArray.length > 0) match {
      case true => DenseMatrix.tabulate(dataSetArray.length,dataSetArray(0).length){(i,j)=>dataSetArray(i)(j)}
      case false => throw new RuntimeException("there is no data")
    }
    val labelVector = origDataSet(::,labelIndex)
    val targetDataSet = origDataSet.delete(labelIndex,Axis._1)
    (targetDataSet,labelVector,origDataSet)
  }

}
