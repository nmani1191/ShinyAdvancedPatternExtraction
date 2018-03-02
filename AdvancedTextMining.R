# This is an R Shiny Application helping to do Pattern Extraction using Text Mining

advanced_pattern_extraction <- function()
{
  #Package Loading
  library(shinydashboard)
  library(shiny)
  library(dplyr)
  library(DT)
  library(stringr)
  library(tm)
  library(RWeka)
  
  #Function Loading
  #######################################################
  Unique_NA_counts <- function(input_data)
  {
    df1 <- data.frame(character(),character(),integer(),integer(),integer(),numeric())
    
    for (name in colnames(input_data)) {
      
      df1 <- rbind(df1,data.frame(ColName=name,Datatype=class(input_data[,name]),Total_Records=nrow(input_data),
                                  Unique_Counts=length(unique(input_data[,name])),
                                  NA_Counts=sum(is.na(input_data[,name])),
                                  NA_Percent=round(sum(is.na(input_data[,name]))/nrow(input_data),2)))
      
    }
    
    df1 <- as.data.frame(df1 %>% arrange(-NA_Counts))
    
    return(df1)
  }
  #######################################################
  Text_Processing<- function (FreeText, wordstoRemove)
  {
    FreeText <- iconv(FreeText, "latin1", "ASCII", sub="")
    wordstoRemove <- iconv(wordstoRemove,"latin1", "ASCII", sub = "")
    
    FreeText <- as.character(FreeText)
    wordstoRemove <- tolower(wordstoRemove)
    FreeText <- gsub("[^ -z]", " ", FreeText)
    FreeText <- gsub("[0-9]", " ", FreeText)
    FreeText <- gsub("[34-47]", " ", FreeText)#Spl character
    FreeText <- stringi::stri_trim(FreeText, side = c("both"))
    FreeText <- tolower(FreeText)
    FreeText <- gsub("[[:punct:]]", " ", FreeText)
    FreeText <- gsub("\\s+", " ", str_trim(FreeText))
    FreeText <- paste0("wordstart ", FreeText, " wordend")
    for (i in 1:length(wordstoRemove)) {
      FreeText <- gsub(paste("*\\b", wordstoRemove[i], "\\b*"), 
                       " ", FreeText)
    }
    FreeText <- gsub("\\s+", " ", str_trim(FreeText))
    FreeText <- substr(FreeText, 10, stringi::stri_length(FreeText) - 
                         8)
    FreeText <- str_trim(FreeText)
    return(FreeText)
  }
  #######################################################
  top5_ngram_generation_idf <- function(analysisdata,ngram)
  {
    #Remove Invalid Characters
    analysisdata <- iconv(analysisdata, "latin1", "ASCII", sub="")
    
    #Replace the NA documents with blank value ""
    Modifiedanalysisdata <- data.frame(analysisdata)
    Modifiedanalysisdata$analysisdata <- as.character(Modifiedanalysisdata$analysisdata)
    Modifiedanalysisdata[is.na(Modifiedanalysisdata)] <- ""
    text_data <- as.character(Modifiedanalysisdata$analysisdata)
    
    #Convert the character into corpus
    corpus.ng<- VCorpus(VectorSource(text_data))
    
    #Use RWeka function to generate Ngrams
    set.seed(3000)
    ngramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = ngram, max = ngram))
    
    #Create Document Term Matrix
    dtm.ngram = DocumentTermMatrix(corpus.ng,control = list(tokenize = ngramTokenizer))
    #Remove sparse terms
    dtm.ngram <- removeSparseTerms(dtm.ngram, 0.99)
    ngramdf <- as.data.frame(as.matrix(dtm.ngram))
    rownames(ngramdf) <- NULL
    
    #Create Term Presence (TP) -> Replace 0 values with NA and > 0 values with 1 in the above DTM
    df <- ngramdf
    df[df==0] <- NA
    df[df>0] <- 1
    output <- df
    
    #Compute IDF -> log(Total no of Documents/No of Documents contains the term)
    id=function(col){sum(!col==0)}
    idf <- log(nrow(ngramdf)/apply(ngramdf, 2, id))
    
    #Repeatable Pattern in Document -> TP*IDF
    for(word in names(idf)){output[,word] <- df[,word] * idf[word]}
    
    #Extract Top Five pattern which has min IDF score (This will be the pattern which occured more frequently)
    row_result <- function(row) {
      top <- ifelse(sum(sort(row) >= 0)>5,5,sum(sort(row) > 0))
      paste(head(names(sort(row)),top),collapse = ",") }
    
    output$Ngram_pattern <- apply(output,1,function(x) row_result(x))
    
    #Replace NA with 0 where we dont have pattern
    output$Ngram_pattern[output$Ngram_pattern==""] <- 0
    
    ngram_pattern <- output$Ngram_pattern
    
    return(ngram_pattern)
  }
  #######################################################
  top5_ngram_generation_tfidf <- function(analysisdata,ngram)
  {
    #Remove Invalid Characters
    analysisdata <- iconv(analysisdata, "latin1", "ASCII", sub="")
    
    #Replace the NA documents with blank value ""
    Modifiedanalysisdata <- data.frame(analysisdata)
    Modifiedanalysisdata$analysisdata <- as.character(Modifiedanalysisdata$analysisdata)
    Modifiedanalysisdata[is.na(Modifiedanalysisdata)] <- ""
    text_data <- as.character(Modifiedanalysisdata$analysisdata)
    
    #Convert the character into corpus
    corpus.ng<- VCorpus(VectorSource(text_data))
    
    #Use RWeka function to generate Ngrams
    set.seed(3000)
    ngramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = ngram, max = ngram))
    
    #Create Document Term Matrix
    dtm.ngram = DocumentTermMatrix(corpus.ng,control = list(tokenize = ngramTokenizer))
    #Remove sparse terms
    dtm.ngram <- removeSparseTerms(dtm.ngram, 0.99)
    ngramdf <- as.data.frame(as.matrix(dtm.ngram))
    rownames(ngramdf) <- NULL
    
    #Create Term Frequency (TF) -> (No of time the Term occurred in the Document/Total no of Terms in the Document)
    df <- ngramdf
    df1 <- as.matrix(df)                 
    tf <- df1/rowSums(df1)
    #tf[is.nan(tf)] <- 0
    
    #Compute IDF -> log(Total no of Documents/No of Documents contains the term)
    id=function(col){sum(!col==0)}
    idf <- log(nrow(df)/apply(df, 2, id))
    
    #Compuute TF IDF -> TF*IDF
    tfidf <- df
    for(word in names(idf)){tfidf[,word] <- tf[,word] * idf[word]}
    
    #Extract Top Five pattern which has max IDF score (This will be the pattern which unique to document)
    row_result <- function(row) {
      top <- ifelse(sum(sort(row,decreasing = T) > 0)>5,5,sum(sort(row,decreasing = T) > 0))
      paste(head(names(sort(row,decreasing = T)),top),collapse = ",") }
    
    tfidf$Ngram_pattern <- apply(tfidf,1,function(x) row_result(x))
    
    #Replace NA with 0 where we dont have pattern
    tfidf$Ngram_pattern[tfidf$Ngram_pattern==""] <- 0
    
    ngram_pattern <- tfidf$Ngram_pattern
    
    return(ngram_pattern)
  }
  #######################################################
  options(shiny.maxRequestSize = 1024*1024^2)
  #Shiny App
  shinyApp (
    
    ui= dashboardPage(skin = "purple",
      dashboardHeader(title = "Advanced Pattern Extraction",titleWidth=300),
      dashboardSidebar(width = 300,
                       conditionalPanel(condition="input.conditionedPanels==1",
                                        fileInput('dataset', 'Choose CSV File',accept=c('.csv')),
                                        actionButton("Validate","validate"),
                                        tags$hr(),
                                        uiOutput("TextVar"),
                                        uiOutput("TechniqueVar"),
                                        uiOutput("Key"),
                                        uiOutput("runbutton"))
      ),
      dashboardBody(
        
        tabsetPanel(id="conditionedPanels",
                    tabPanel("Outputs",value = 1,
                             fluidRow(column(width=12,h4("Data Summary",style="color:darkgreen"),DT::dataTableOutput("dataSummary"))),
                             uiOutput("dataSummary_download"),
                             tags$hr(),
                             uiOutput("Pattern_download")
                    )
        )
      )
    ),
    server= function(input, output, session) {
      
      observeEvent(input$Validate,{
        
        input$Validate # Re-run when button is clicked
        
        withProgress(message = 'Validating...', value = 0, {
          
          incProgress(0.25, detail = " 25%")
          
          inFile <- input$dataset
          ins_data_set <- read.csv(inFile$datapath,header = T,strip.white = T,
                                   na.strings = c(""," ","NA","NULL","na","null"),fileEncoding = "latin1")
          Column_names <- colnames(ins_data_set)
          
          output$TextVar <- renderUI({
            selectInput("textField", label = "Select Text Field for mining:",  c("--select--", Column_names))
          })
          
          output$TechniqueVar <- renderUI({
            selectInput("techField", label = "Select Text Mining Method:",  c("IDF", "TFIDF"))
          })
          
          output$Key <- renderUI({
            fileInput('keyword', 'Upload Keywords to remove(CSV File)',accept=c('.csv'))
          })
          
          output$runbutton <- renderUI({
            actionButton("run","Run")
          })
          
          incProgress(0.5, detail = " 50%")
          
          dataSummary <- reactive({
            
            validate(
              need(input$Validate != 0, "Please Upload Date & Validate")
            )
            
            isolate({
              summary<- Unique_NA_counts(ins_data_set)
              return(summary)
            })
          })
          
          incProgress(0.75, detail = " 75%")
          
          output$dataSummary<- DT::renderDataTable((datatable(dataSummary())),filter='top',options=list(autoWidth=TRUE))
          
          output$dataSummary_download <- renderUI({
            fluidRow(
              column(width=6,h4("Download Data Summary Table",style="color:darkgreen")),
              column(width=3,downloadButton('dataSummary_downloader',"Download Table"))
            )
          })
          
          output$dataSummary_downloader <- downloadHandler(
            filename = "DataSummary.csv",
            content = function(file) {
              write.csv(dataSummary(), file,row.names = F) })
          
          incProgress(1, detail = " 100%")
        })
      })
      
      observeEvent(input$run,{
        
        input$run
        
        withProgress(message = 'Processing...', value = 0, {
          
          incProgress(0.05, detail = " 5%")
          
          inFile <- input$dataset
          ins_data_set <- read.csv(inFile$datapath,header = T,strip.white = T,
                                   na.strings = c(""," ","NA","NULL","na","null"),fileEncoding = "latin1")
          Column_names <- colnames(ins_data_set)
          
          incProgress(0.1, detail = " 10%")
          
          
          output$Pattern_download <- renderUI({
            fluidRow(
              column(width=6,h4("Download Data with Pattern File",style="color:darkgreen")),
              column(width=3,downloadButton('dataPattern_download',"Download Data"))
            )
          })
          
          incProgress(1, detail = " 100%")
          
          ngram_output <- reactive({
            
            validate(
              need(input$run != 0,
                   (!input$techField %in% c("")),"Please Upload Data & Run it")
            )
            isolate({
              withProgress(message = 'Processing...', value = 0, {
                
                inFileKey <- input$keyword
                if(!is.null(inFileKey))
                {
                  keywords <- read.csv(inFileKey$datapath,header = T,strip.white = T,fileEncoding = "latin1")
                  keywords <- as.character( keywords[,1])
                  keywords <- c(stopwords("english"), keywords)
                  Keywordstoremove <- as.data.frame(keywords)
                  colnames(Keywordstoremove) <- "Keyword"
                }
                else
                {
                  Keywordstoremove <- as.data.frame(stopwords("english"))
                  colnames(Keywordstoremove ) <- c("Keyword")
                }
                
                incProgress(0.1, detail = " 10%")
                total_rows <- nrow(ins_data_set)
                
                Textcolumn <- input$textField
                preproccessdata <- Text_Processing(ins_data_set[,which(colnames(ins_data_set)==Textcolumn)],Keywordstoremove$Keyword)
                
                incProgress(0.25, detail = " 25%")
                
                if(input$techField=="IDF")
                {
                  ins_data_set$Unigram <- top5_ngram_generation_idf(preproccessdata,1)
                } else {
                  ins_data_set$Unigram <- top5_ngram_generation_tfidf(preproccessdata,1)
                }
                
                incProgress(0.5, detail = " 50%")
                
                if(input$techField=="IDF")
                {
                  ins_data_set$Bigram <- top5_ngram_generation_idf(preproccessdata,2)
                } else {
                  ins_data_set$Bigram <- top5_ngram_generation_tfidf(preproccessdata,2)
                }
                
                incProgress(0.75, detail = " 75%")
                
                if(input$techField=="IDF")
                {
                  ins_data_set$Trigram <- top5_ngram_generation_idf(preproccessdata,3)
                }else {
                  ins_data_set$Trigram <- top5_ngram_generation_tfidf(preproccessdata,3)
                }
                
                return(list(dataset_pattern=ins_data_set))
                
                incProgress(1, detail = " 100%")
                
              })
            })
          })
          
          output$dataPattern_download <- downloadHandler(
            filename = "DataWithAdvancedPattern.csv",
            content = function(file) {
              write.csv(ngram_output()$dataset_pattern, file,row.names = F) })
          
        })
      })
      
    }
  )
  
}
