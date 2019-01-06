package ml.method

import breeze.linalg.{DenseMatrix, DenseVector}
import play.api.libs.json.{JsObject, JsValue, Json}

class DecisionTree(
                    protected val dataSet:DenseMatrix[String],
                    protected val labels:DenseVector[String]) {

  // 计算香农熵
  private def getShannonEnt(lb:DenseVector[String]) = {
    var map = Map[String,Int]()
    var ent:Double = 0
    lb.foreach(label=>{
      map = map.updated(label,map.getOrElse(label,0)+1)
    })
    map.foreach(pair=>{
      val prob = pair._2.toDouble / lb.length.toDouble
      ent -= prob * (Math.log(prob) / Math.log(2))
    })
    ent
  }

  // 根据特征下标索引和特征值切分数据集
  private def splitDataSet(ds:DenseMatrix[String],lb:DenseVector[String],axis:Int,value:String) = {
    var retDataSeq = Seq[Seq[String]]()
    var retLabelSeq = Seq[String]()
    (0 to (ds.rows-1)).foreach(row=>{
      if(ds.valueAt(row,axis) == value){
        var rowDataSeq = Seq[String]()
        (0 to (axis-1)).foreach(col=>{
          rowDataSeq = rowDataSeq :+ ds.valueAt(row,col)
        })
        ((axis+1) to (ds.cols-1)).foreach(col=>{
          rowDataSeq = rowDataSeq :+ ds.valueAt(row,col)
        })
        retDataSeq = retDataSeq :+ rowDataSeq
        retLabelSeq = retLabelSeq :+ lb(row)
      }
    })
    (DenseMatrix.tabulate(retDataSeq.length,retDataSeq(0).length)((r,c)=>{retDataSeq(r)(c)}),
      DenseVector.tabulate(retLabelSeq.length){i=>retLabelSeq(i)})
  }

  // 根据数据集计算最优切分特征
  private def chooseBestFeature(ds:DenseMatrix[String],lb:DenseVector[String]) = {
    val baseEnt = getShannonEnt(lb)
    var bestInfoGain:Double = 0
    var bestFeature:Int = -1
    (0 until ds.cols).foreach{col=>
      val newEnt:Double = ds(::,col).toArray.toSet.foldLeft[Double](0){(newEnt,x)=>
        val (subDs,subLb) = splitDataSet(ds,lb,col,x)
        newEnt + (subDs.rows.toDouble / ds.rows.toDouble) * getShannonEnt(subLb)
      }
      if(baseEnt - newEnt > bestInfoGain){
        bestInfoGain = baseEnt - newEnt
        bestFeature = col
      }
    }
    bestFeature
  }

  // 投票法确定分类
  private def majorityCnt(lb:DenseVector[String]) = lb.foldLeft(Map()){(cntMap,lab)=>
    cntMap.asInstanceOf[Map[String,Int]].updated(lab,cntMap.asInstanceOf[Map[String,Int]].getOrElse(lab,0)+1)
  }.asInstanceOf[Map[String,Int]].toSeq.sortWith((a,b)=>{b._2<a._2})(0)._1

  // 创建决策树
  private def createTree(ds:DenseMatrix[String],lb:DenseVector[String],featureNames:Seq[String]):JsValue = {
    if(lb.toArray.toSet.size == 1) Json.toJson(lb(0))
    else if(ds.cols == 0) Json.toJson(majorityCnt(lb))
    else{
      val bestFeature = chooseBestFeature(ds,lb)
      val bestFeatureName = featureNames(bestFeature)
      var subTree = Json.obj()
      ds(::,bestFeature).toArray.toSet.foreach((x:String)=>{
        val (subDs,subLb) = splitDataSet(ds,lb,bestFeature,x)
        subTree = subTree + (x -> createTree(subDs,subLb,featureNames.filter(_!=bestFeatureName)))
      })
      Json.obj(bestFeatureName -> subTree)
    }
  }

  private def doClassify(in:DenseVector[String],inputTree:JsObject):String = {
    val featureName = inputTree.keys.toSeq(0)
    val featureVal = in(featureName.toInt)
    (inputTree \ featureName \ featureVal).validate[JsObject].asOpt match {
      case Some(subTree) => doClassify(in,subTree)
      case None => (inputTree \ featureName \ featureVal).as[String]
    }
  }

  val tree = createTree(dataSet,labels,(0 until dataSet.cols).map(_.toString))

  def classify(in:DenseVector[String]) = {
    doClassify(in,tree.asInstanceOf[JsObject])
  }
}
