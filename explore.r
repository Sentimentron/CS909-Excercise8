# Must have the XML library installed
require('XML')
require('openNLP')

# Reads the source document set
readXmlDocFormat <- function() {

  doc <- xmlTreeParse('converted.xml')
  r <- xmlRoot(doc)

  initialDf <- data.frame(stringsAsFactors=FALSE)

  # Pull out text and title first...
  for (child in xmlChildren(r)) {

      text <- ""
      title <- ""

      for (node in xmlChildren(child)) {
	  nodeName <- xmlName(node)
	  if (nodeName == "Text") {
	      text <- xmlValue(node)
	  }
	  if (nodeName == "Title") {
	      title <- xmlValue(node)
	  }
      }

      # Then assign each topic
      for (node in xmlChildren(child)) {

	  if (xmlName(node) != "Topics") next()

	  for (topicNode in xmlChildren(node)) {
	      topic <- xmlValue(topicNode)
	      newRow <- c(text, title, topic)
	      # Each topic appears as a separate row in the dataframe
	      initialDf <- rbind(initialDf, newRow)
	  }

      }

  }

  # Convert topics to a more compact representation
  initialDf$topic <- as.factor(initialDf$topic)
  return(initialDf)
}


getTopTopics <- function(srcData) {
  counts <- table(srcData$topic)
  counts <- sort(counts, decreasing = TRUE)
  return (counts[1:10])
}

removeExtraneousTopics <- function(srcData) {
  topics <- getTopTopics(srcData);
  filteredDf <- data.frame()
  for (topic in names(topics)) {
    subsetDf <- srcData[srcData$topic == topic,]
    filteredDf <- rbind(subsetDf, filteredDf)
  }
  return(filteredDf);
}

annotateStrWithPOSTags <- function(s) {

  sent_token_annotator <- Maxent_Sent_Token_Annotator()
  word_token_annotator <- Maxent_Word_Token_Annotator()
  a2 <- annotate(s, list(sent_token_annotator, word_token_annotator))
  pos_tag_annotator <- Maxent_POS_Tag_Annotator()
  a3 <- annotate(s, pos_tag_annotator, a2)
  a3w <- subset(a3, type=="word")
  outputStr <- list()
  for (rowNo in seq(1, length(a3w))) {
    feature <- a3w[rowNo]
    tag <- as.String(feature$features[[1]])
    start <- feature$start
    end <- feature$end
    word <- substr(s,start,end)
    if (word != tag) {
      word <- sprintf("%s/%s", word, tag)
    }
    outputStr <- rbind(outputStr, word)
  }
  return(paste(outputStr, seperator=" "))

}

tokenizeStrToWords <- function(s, word_token_annotator, sent_token_annotator) {
  outputStr <- list()
  sent_token_annotator <- Maxent_Sent_Token_Annotator()

  a2 <- annotate(s, list(sent_token_annotator, word_token_annotator))
  a2w <- subset(a2, type=="word")
  for (rowNo in seq(1, length(a2w))) {
    feature <- a2w[rowNo]
    start <- feature$start
    end <- feature$end
    word <- substr(s,start,end)
    outputStr <- rbind(outputStr, word)
  }
  return (paste(outputStr))
}

buildUnigramFeatureSpace <- function(df) {

  outputDf <- data.frame()
  word_token_annotator <- Maxent_Word_Token_Annotator()

  for (i in seq(1, nrow(df))) {
    print(sprintf("%d/%d",i,nrow(df)))
    s <- df[i, "text"]
    words <- tokenizeStrToWords(s, sent_token_annotator, word_token_annotator)
    for (word in words) {
      outputDf[i, word] <- TRUE
    }
    outputDf[i, "topicAttr"] <- df[i, "topic"]
  }

  outputDf[is.na(outputDf)] <- FALSE

  return(outputDf)

}

initialDf <- readXmlDocFormat()
filteredDf <- removeExtraneousTopics(initialDf)
