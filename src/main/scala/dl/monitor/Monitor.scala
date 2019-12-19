package dl.monitor

trait Monitor {

  def eachEpoch(epochCount:Int):Unit = {}

  def eachIter(iterCount:Int):Unit = {}

}
