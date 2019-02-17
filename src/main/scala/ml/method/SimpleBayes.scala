package ml.method

import breeze.linalg._

object SimpleBayes{
  object Model {
    val CJ:Byte = 0
    val CD:Byte = 1
  }
}
class SimpleBayes(val data:Seq[Seq[String]],val labels:Seq[String],val model:Byte = SimpleBayes.Model.CD) {

  private val dataItemSetList = data.foldLeft(Set[String]()){(set,row) => set ++ row.toSet}.toSeq

  private def inputItems2ExistList(inputItems:Seq[String]) = {
    inputItems.foldLeft(Seq.fill(dataItemSetList.length)(0)){(existList,item)=>
      val idx = dataItemSetList.indexOf(item)
      if(idx > -1) {
        if(model == SimpleBayes.Model.CJ) existList.updated(idx,1)
        else existList.updated(idx,existList(idx)+1)
      }
      else existList
    }
  }

  private def trainData() = {
    val transedData = data.map(inputItems2ExistList(_))
    val dataMat = DenseMatrix.tabulate(transedData.length,transedData(0).length){(x,y)=>{transedData(x)(y)}}
    val labelsVec = DenseVector(labels.toArray)
    labels.toSet.foldLeft[Map[String,Tuple2[Double,DenseVector[Double]]]](Map()){(map,lab)=>
      val pType = labels.filter(_ == lab).length.toDouble / dataMat.rows.toDouble
      var allItemsCount:Double = 2
      var labItemsCountVec = DenseVector.ones[Double](dataMat.cols)
      (0 until dataMat.rows).foreach{i=>
        if(lab == labelsVec(i)){
          val dataRowVec = DenseVector.tabulate(dataMat.cols){x=>dataMat.valueAt(i,x)}
          allItemsCount += sum(dataRowVec)
          labItemsCountVec = labItemsCountVec + dataRowVec.map(_.toDouble)
        }
      }
      labItemsCountVec = (labItemsCountVec /:/ allItemsCount).map(Math.log(_))
      map.updated(lab,(pType,labItemsCountVec))
    }
  }

  private val trainedDataResult = trainData()

  def classify(input:Seq[String]) = {
    val inputVec = DenseVector(inputItems2ExistList(input).map(_.toDouble).toArray)
    trainedDataResult.map(x=>{
      (x._1,sum(inputVec *:* x._2._2) + Math.log(x._2._1))
    }).reduce((a,b)=>{
      if(a._2 > b._2) a
      else b
    })._1
  }

}
