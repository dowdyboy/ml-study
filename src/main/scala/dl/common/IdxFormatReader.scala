package dl.common

import java.io._
import java.util

case class IdxData(dataType:Byte,dimensionsSize:Seq[Int],data:Seq[_])

object IdxFormatReader {

  var BUFFER_SIZE = 4096000

  object DataType {
    val U_BYTE:Byte = 0x08.toByte
    val BYTE:Byte = 0x09.toByte
    val SHORT:Byte = 0x0B.toByte
    val INT:Byte = 0x0C.toByte
    val FLOAT:Byte = 0x0D.toByte
    val DOUBLE:Byte = 0x0E.toByte
  }

  private def byteSeq(is:InputStream) = {
    val dataArr = new Array[Byte](BUFFER_SIZE)
    var dataSeq = scala.collection.mutable.Seq[Byte]()
    var len = is.read(dataArr,0,dataArr.length)
    while(len > 0){
      if(dataArr.length == len){
        dataSeq = dataSeq ++ dataArr
      }else{
        dataSeq = dataSeq ++ util.Arrays.copyOfRange(dataArr,0,len)
      }
      len = is.read(dataArr,0,dataArr.length)
    }
    dataSeq
  }

  private def shortSeq(is:InputStream) = {
    val dataArr = new Array[Byte](BUFFER_SIZE)
    var dataSeq = scala.collection.mutable.Seq[Short]()
    var len = is.read(dataArr,0,dataArr.length)
    while(len > 0){
      (0 until len by 2).foreach{i=>
        val a = (dataArr(i) & 0xff) << 8
        val b = (dataArr(i+1) & 0xff)
        val value:Short = (a | b).toShort
        dataSeq = dataSeq :+ value
      }
      len = is.read(dataArr,0,dataArr.length)
    }
    dataSeq
  }

  private def intSeq(is:InputStream) = {
    val dataArr = new Array[Byte](BUFFER_SIZE)
    var dataSeq = scala.collection.mutable.Seq[Int]()
    var len = is.read(dataArr,0,dataArr.length)
    while(len > 0){
      (0 until len by 4).foreach{i=>
        val a = (dataArr(i) & 0xff) << 24
        val b = (dataArr(i+1) & 0xff) << 16
        val c = (dataArr(i+2) & 0xff) << 8
        val d = (dataArr(i+3) & 0xff)
        val value:Int = (a | b | c | d)
        dataSeq = dataSeq :+ value
      }
      len = is.read(dataArr,0,dataArr.length)
    }
    dataSeq
  }

  private def floatSeq(is:InputStream) = {
    val dataArr = new Array[Byte](BUFFER_SIZE)
    var dataSeq = scala.collection.mutable.Seq[Float]()
    var len = is.read(dataArr,0,dataArr.length)
    while(len > 0){
      (0 until len by 4).foreach{i=>
        val a = (dataArr(i) & 0xff) << 24
        val b = (dataArr(i+1) & 0xff) << 16
        val c = (dataArr(i+2) & 0xff) << 8
        val d = (dataArr(i+3) & 0xff)
        val value:Float = (a | b | c | d)
        dataSeq = dataSeq :+ value
      }
      len = is.read(dataArr,0,dataArr.length)
    }
    dataSeq
  }

  private def doubleSeq(is:InputStream) = {
    val dataArr = new Array[Byte](BUFFER_SIZE)
    var dataSeq = scala.collection.mutable.Seq[Double]()
    var len = is.read(dataArr,0,dataArr.length)
    while(len > 0){
      (0 until len by 8).foreach{i=>
        val a = (dataArr(i) & 0xff.toLong) << 56
        val b = (dataArr(i+1) & 0xff.toLong) << 48
        val c = (dataArr(i+2) & 0xff.toLong) << 40
        val d = (dataArr(i+3) & 0xff.toLong) << 32
        val e = (dataArr(i+4) & 0xff.toLong) << 24
        val f = (dataArr(i+5) & 0xff.toLong) << 16
        val g = (dataArr(i+6) & 0xff.toLong) << 8
        val h = (dataArr(i+7) & 0xff.toLong)
        val value:Double = (a | b | c | d | e | f | g | h)
        dataSeq = dataSeq :+ value
      }
      len = is.read(dataArr,0,dataArr.length)
    }
    dataSeq
  }

  def fromFile(path:String) = {
    val file = new File(path)
    val is = new BufferedInputStream(new FileInputStream(file))
    val magicNumber = new Array[Byte](4)
    is.read(magicNumber,0,4)
    val dataType:Byte = magicNumber(2)
    val dimensionCount:Int = magicNumber(3)
    val dimensionsSize:Seq[Int] = (0 until dimensionCount).map{i=>
      val sizeArr = new Array[Byte](4)
      is.read(sizeArr,0,4)
      val a = (sizeArr(0) & 0xff) << 24
      val b = (sizeArr(1) & 0xff) << 16
      val c = (sizeArr(2) & 0xff) << 8
      val d = (sizeArr(3) & 0xff)
      a | b | c | d
    }
    val data = dataType match {
      case DataType.U_BYTE => byteSeq(is).map(x=>{x & 0xff})
      case DataType.BYTE => byteSeq(is)
      case DataType.SHORT => shortSeq(is)
      case DataType.INT => intSeq(is)
      case DataType.FLOAT => floatSeq(is)
      case DataType.DOUBLE => doubleSeq(is)
      case _ => throw new RuntimeException("unsupport this type of data in idx format")
    }
    is.close()
    IdxData(dataType,dimensionsSize,data)
  }

}
