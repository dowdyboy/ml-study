package ml

import java.io.{BufferedWriter, File, FileOutputStream, OutputStreamWriter}

import breeze.linalg.DenseVector
import ml.common.{TsvSplitType, Utils}
import ml.method.{DecisionTree, Knn, SimpleBayes}
import play.api.libs.json.Json

import scala.io.Source

object Main {

  // knn算法分类约会信息
  def knnDatingClassify() = {
    val (dataSet,label,_) = Utils.createDataSet("assets/knn/datingTrainSet.txt",4,3,TsvSplitType)
    val knn = new Knn(dataSet.map(_.toDouble),label,10)
    val (testDataSet,testLabel,_) = Utils.createDataSet("assets/knn/datingTestSet.txt",4,3,TsvSplitType)
    var rightCount = 0
    for(i <- 0 to (testDataSet.rows-1)){
      val classifyLabel = knn.classify(testDataSet(i,::).inner.map(_.toDouble))
      val realLabel = testLabel(i)
      if(classifyLabel == realLabel) rightCount += 1
      println(s"classify label is ${classifyLabel} , real label is ${realLabel}")
    }
    println(s"the right percent is ${rightCount.toDouble / testDataSet.rows.toDouble * 100}%")
  }

  // knn算法分类约会信息（使用归一化特征量）
  def knnDatingClassifyWithNorm() = {
    val (dataSet,label,_) = Utils.createDataSet("assets/knn/datingTrainSet.txt",4,3,TsvSplitType)
    val (normDataSet,normRangeVect,normMinVect) = Utils.norm(dataSet.map(_.toDouble))
    val knn = new Knn(normDataSet,label,10)
    val (testDataSet,testLabel,_) = Utils.createDataSet("assets/knn/datingTestSet.txt",4,3,TsvSplitType)
    var rightCount = 0
    for(i <- 0 to (testDataSet.rows-1)){
      val classifyLabel = knn.classify(Utils.norm(testDataSet(i,::).inner.map(_.toDouble),normMinVect,normRangeVect))
      val realLabel = testLabel(i)
      if(classifyLabel == realLabel) rightCount += 1
      println(s"classify label is ${classifyLabel} , real label is ${realLabel}")
    }
    println(s"the right percent is ${rightCount.toDouble / testDataSet.rows.toDouble * 100}%")
  }

  // 合并数字文件
  def createDigitsSingleFile() = {
    val dir = new File("assets/knn/testDigits")
    val outFile = new File("assets/knn/testDigits.txt")
    val bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outFile)))
    dir.listFiles().foreach{file =>
      val number = file.getName.split("_")(0)
      Source.fromFile(file).getLines().foreach{line =>
        line.toCharArray.foreach{c =>
          bw.write(s"${c},")
        }
      }
      bw.write(s"${number}\n")
      bw.flush()
    }
    bw.close()
  }

  // KNN算法分类数字数据
  def knnDigitsClassify() = {
    val (dataSet,label,_) = Utils.createDataSet("assets/knn/trainingDigits.txt",1025,1024)
    val knn = new Knn(dataSet.map(_.toDouble),label,10)
    val (testDataSet,testLabel,_) = Utils.createDataSet("assets/knn/testDigits.txt",1025,1024)
    var rightCount = 0
    for(i <- 0 to (testDataSet.rows - 1)){
      val classifyLabel = knn.classify(testDataSet(i,::).inner.map(_.toDouble))
      val realLabel = testLabel(i)
      if(classifyLabel == realLabel) rightCount += 1
      println(s"classify label is ${classifyLabel} , real label is ${realLabel}")
    }
    println(s"the right percent is ${rightCount.toDouble / testDataSet.rows.toDouble * 100}%")
  }

  // 决策树ID3算法测试--分类是否海洋动物
  def decisionTreeOceanTestClassify = {
    val (ds,lb,_) = Utils.createDataSet("assets/decision_tree/ocean_test.txt",3,2)
    val dt = new DecisionTree(ds,lb)
    println(dt.classify(DenseVector("1","1")))
  }

  // 决策树ID3算法分类眼镜类型
  def decisionTreeLensesClassify = {
    val (ds,lb,_) = Utils.createDataSet("assets/decision_tree/lenses.txt",5,4,TsvSplitType)
    val dt = new DecisionTree(ds,lb)
    println(dt.classify(DenseVector("pre","myope","yes","reduced")))
  }

  // 朴素贝叶斯分类侮辱性英文句子
  def simpleBayesStupidWords = {
    val data = Seq(
      Seq("my","dog","has","flea","problems","help","please"),
      Seq("maybe","not","take","him","to","dog","park","stupid"),
      Seq("my","dalmation","is","so","cute","I","love","him"),
      Seq("stop","posting","stupid","worthless","garbage"),
      Seq("mr","licks","ate","my","steak","how","to","stop","him"),
      Seq("quit","buying","worthless","dog","food","stupid")
    )
    val label = Seq("0","1","0","1","0","1")
    val sb = new SimpleBayes(data,label)
    println(sb.classify(Seq("love","my","dalmation")))
    println(sb.classify(Seq("stupid","garbage")))
  }

  // 朴素贝叶斯分类垃圾邮件
  def simpleBayesGarbageEmails = {
    val d = Utils.SimpleBayes.readEmailFile
    val sb = new SimpleBayes(d._1,d._2)
    (0 until d._3.length).foreach(i=>{
      println(s"classify label : ${sb.classify(d._3(i))} , real Label : ${d._4(i)}")
    })
  }

  def main(args: Array[String]): Unit = {
    (0 to 100).foreach(i=>{simpleBayesGarbageEmails})
  }

}
