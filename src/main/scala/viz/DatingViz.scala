package viz

import breeze.linalg._
import breeze.plot._
import ml.common.{TsvSplitType, Utils}

object DatingViz {

  def test = {
    val f = Figure()
    val p = f.subplot(0)
    val x = linspace(0,10)
    val g = breeze.stats.distributions.Gaussian(0,1)
    p.title = "Test Pic"
    p.xlabel = "x axis"
    p.ylabel = "y axis"
    p += plot(x,x.map(x=>{x*x}))
    p += scatter(x,x.map(x=>{x*x*x}),(i:Int)=>{0.1},(i:Int)=>{PaintScale.blue})
    p += image(DenseMatrix.rand(1000,1000))
    f.saveas("test.png")
  }

  def datingDataScatter = {
    val (_,_,dataset) = Utils.createDataSet("assets/knn/datingTrainSet.txt",4,3,TsvSplitType)
    var largeDosesFlyMile = Seq[Double]()
    var largeDosesGameTime = Seq[Double]()
    var didntLikeFlyMile = Seq[Double]()
    var didntLikeGameTime = Seq[Double]()
    (0 to (dataset.rows-1)).foreach(i => {
      dataset(i,3) match {
        case "largeDoses" =>
          largeDosesFlyMile = largeDosesFlyMile :+ dataset(i,0).toDouble
          largeDosesGameTime = largeDosesGameTime :+ dataset(i,1).toDouble
        case "didntLike" =>
          didntLikeFlyMile = didntLikeFlyMile :+ dataset(i,0).toDouble
          didntLikeGameTime = didntLikeGameTime :+ dataset(i,1).toDouble
        case _=>
      }
    })
    val f = Figure()
    val p = f.subplot(0)
    p.title = "约会偏好数据（红色为“不喜欢”，蓝色为“喜欢”）"
    p.xlabel = "每年乘坐飞机的里程数"
    p.ylabel = "每周玩游戏时间所占比例"
    p += scatter(
      new DenseVector[Double](largeDosesFlyMile.toArray),
      new DenseVector[Double](largeDosesGameTime.toArray),
      (i:Int)=>{1000},
      (i:Int)=>{PaintScale.blue})
    p += scatter(
      new DenseVector[Double](didntLikeFlyMile.toArray),
      new DenseVector[Double](didntLikeGameTime.toArray),
      (i:Int)=>{1000},
      (i:Int)=>{PaintScale.red})
    f.saveas("datingDataScatter.png")
  }

}
