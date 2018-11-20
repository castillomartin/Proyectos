#include "primeFinder.h"

int main(int argc, char *argv[]) {
  clock_t begin = clock();
  // /* PROGRAM DON'T DELETE
  pthread_mutex_init(&mutex, NULL);

  //                                                                     STEP 1
  //--------------------------------------------------------------------------->
  if (validateCommandLine(argc,argv) == FAIL){
    pthread_mutex_destroy(&mutex);
    return 0;
  }

  //                                                                     STEP 2
  //--------------------------------------------------------------------------->
  int nthPrime = atoi(argv[1]);
  int NUMBER_OF_THREADS = atoi(argv[2]);
  int chunkSize = atoi(argv[3]);

  //                                                                     STEP 3
  //--------------------------------------------------------------------------->
  int ARRAY_LENGTH = nthPrime + 1;
  globalPrimeArray = malloc(ARRAY_LENGTH * sizeof(*globalPrimeArray));
  int value = 0;
  for (size_t arrayIndex = 0; arrayIndex < ARRAY_LENGTH; arrayIndex++) {
    globalPrimeArray[arrayIndex] = value;
    value +=1;
  }

  // 0 and 1 are not considered prime numbers so then set them to 0
  globalPrimeArray[0] = 0;
  globalPrimeArray[1] = 0;
  /************************** [  INSERT 1  ] *********************************/

  //                                                                     STEP 4
  //--------------------------------------------------------------------------->
  // Create the work packets
  int NUMBER_OF_WORK_PACKETS = (int)ceil((double)ARRAY_LENGTH/(double)chunkSize);
  workSegment = NULL;
  createWork(&workSegment, ARRAY_LENGTH, NUMBER_OF_WORK_PACKETS, chunkSize);
  /************************** [  INSERT 2  ] *********************************/

  //                                                                   STEP 4-5
  //--------------------------------------------------------------------------->
  // Variables that will be used by each thread
  UP_TO_THIS_NUMBER = (int)sqrt((double)(nthPrime));
  kthValueON = 2; // First Prime is 2
  packetON = 0;
  packetsLEFT = NUMBER_OF_WORK_PACKETS; // will be meant to decrease every time the method getWork is called.
  resetPACKETS = NUMBER_OF_WORK_PACKETS;

  //                                                                     STEP 5
  //--------------------------------------------------------------------------->
  struct threadStruct* threadPTR = (struct threadStruct*)malloc(NUMBER_OF_THREADS*sizeof(struct threadStruct));
  for (size_t i = 0; i < NUMBER_OF_THREADS; i++) {
    threadPTR[i].threadIndex = i;
    pthread_create(&threadPTR[i].id, NULL, workMethod, &threadPTR[i]);
  }
  // JOIN THE THREADS THAT WHERE CREATED
  for (size_t i = 0; i < NUMBER_OF_THREADS; i++) {
    pthread_join(threadPTR[i].id, NULL);
  }
  pthread_mutex_destroy(&mutex);
  clock_t end = clock();

  //                                                                 FINAL STEP
  //--------------------------------------------------------------------------->

  // Displaying only the prime numbers globalPrimeArray
  printf("\nTHESE ARE THE PRIME NUMBERS FROM 0 -> %d\n", nthPrime);
  for (size_t arrayIndex = 0; arrayIndex < ARRAY_LENGTH ; arrayIndex++) {
    if (globalPrimeArray[arrayIndex] == 0){
      // DO NOTHING
    } else {
      printf("%d ", globalPrimeArray[arrayIndex]);
    }
  }

  double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("\n\nTIME THE PROGRAM TOOK: %f\n", time_spent);
  if(globalPrimeArray[nthPrime] == 0){
    printf("nth Number: %d%s\n", nthPrime, " [NOT PRIME NUMBER]");
  } else {
    printf("nth Number: %d%s\n", nthPrime , " [PRIME NUMBER]");
  }
  printf("%s\n", "");
  return 0;
  // */

} // END OF MAIN

int validateCommandLine(int argc, char *argv[]){
  if (argc != 4) {
    printf("%s\n", "Enter the correct number of arguments");
    return FAIL; // FAILED THE TEST
  } else if (checkForOnlyNumbers(argc,argv) == FAIL){
    return FAIL; // FAILED THE TEST
  } else if(secondParameter(argv) == FAIL){
    return FAIL;
  } else if(thirdParameter(argv) == FAIL){
    return FAIL;
  }
  return PASS; // PASSED THE TEST
}

int checkForOnlyNumbers(int argc, char * argv[]) {
  // ********* CHECK IF THERE IS ANYTHING INPUTED BESIDES NUMBERS *************
  size_t row, column;
  for(row = 1; row < argc; row++){
    for(column = 0; argv[row][column] != '\0'; column++){
      int valueOfArgv = argv[row][column];
      if (valueOfArgv < ZERO_ASCII || valueOfArgv > NINE_ASCII ) { //57 is 9 and 48 is 0 in ascii
        printf("\n%s\n", "***************************************************");
        printf("%s\n%s\n", "There was an incorrect input value. Only positive",
        "whole values are accepted.");
        printf("\n");
        printf("%s\n\n", "***************************************************");
        return FAIL; // didn't work is the number 0
      }
    }
  }
  return PASS; // if it worked then return 1
}

int secondParameter(char * argv[]) {
  if (atoi(argv[2]) == 0){
    printf("\n%s\n\n", "This wont do anything. Change the second parameter so that it can be at least 1.");
    return FAIL;
  }
  return PASS;
}

int thirdParameter(char * argv[]) {
  if (atoi(argv[3]) == 0){
    printf("\n%s\n\n", "This wont do anything. Change the third parameter so that it can be at least 1.");
    return FAIL;
  }
  return PASS;
}


void createWork(struct work **returnWorkPackets, int ARRAY_LENGTH, int NUMBER_OF_WORK_PACKETS, int chunkSize) {
  struct work* tempWorkArray = (struct work*)malloc(NUMBER_OF_WORK_PACKETS*sizeof(struct work));

  int minValueChecking, maxValueChecking, numbersLeftToCheck;
  numbersLeftToCheck = ARRAY_LENGTH;

  // Initialize the work struct so we can add each work workSegment item into the queue.
  for(size_t packetON = 0; packetON < NUMBER_OF_WORK_PACKETS; packetON++){
    if(packetON == 0){ // Initializing the first workSegment
      minValueChecking = 0;
      maxValueChecking = chunkSize - 1;  // This is to fix the offset of arrays
      tempWorkArray[packetON].workMin = minValueChecking;
      tempWorkArray[packetON].workMax = maxValueChecking;
      // Since you already gave chunkSize subtract it from remainingNumbers
      numbersLeftToCheck -= chunkSize;
    } else if (packetON == (NUMBER_OF_WORK_PACKETS - 1)) { // Initializing the last workSegment
      minValueChecking = maxValueChecking + 1;
      maxValueChecking = maxValueChecking + numbersLeftToCheck;
      tempWorkArray[packetON].workMin = minValueChecking;
      tempWorkArray[packetON].workMax = maxValueChecking;
    } else { // Initializing any workSegment in between
      minValueChecking = maxValueChecking + 1;
      maxValueChecking = maxValueChecking + chunkSize;
      tempWorkArray[packetON].workMin = minValueChecking;
      tempWorkArray[packetON].workMax = maxValueChecking;
      numbersLeftToCheck -= chunkSize;
    }
  }

  *returnWorkPackets = tempWorkArray;
}

void getWork(struct work **returnWorkPacket) {
  int ONE_WORK_PACKET = 1;
  struct work* tempWorkArray = (struct work*)malloc(ONE_WORK_PACKET*sizeof(struct work));

  if(packetsLEFT == 0){
    kthValueON += 1;
    packetON = 0;  // RESET VALUES
    packetsLEFT = resetPACKETS; // RESET VALUES

    tempWorkArray->workMin = workSegment[packetON].workMin;
    tempWorkArray->workMax = workSegment[packetON].workMax;
    tempWorkArray->kthValue = kthValueON;
    packetON += 1;
    packetsLEFT -= 1;
  } else {
    tempWorkArray->workMin = workSegment[packetON].workMin;
    tempWorkArray->workMax = workSegment[packetON].workMax;
    tempWorkArray->kthValue = kthValueON;
    packetON += 1;
    packetsLEFT -= 1;
  }

  *returnWorkPacket = tempWorkArray;
}

void* workMethod(void* p){
  struct threadStruct* individualStruct = (struct threadStruct*)p;
  struct work* individualWorkPacket;

  // int kthValueON = determineKthValue(int arrayIndex, int arrLength);
  while (kthValueON <= UP_TO_THIS_NUMBER) {
    pthread_mutex_lock(&mutex);
    getWork(&individualWorkPacket);
    individualStruct->blockNum.workMin = individualWorkPacket->workMin;
    individualStruct->blockNum.workMax = individualWorkPacket->workMax;
    individualStruct->blockNum.kthValue = individualWorkPacket->kthValue;

    /************************** [  INSERT 3  ] *******************************/

    int workMin = individualStruct->blockNum.workMin;
    int workMax = individualStruct->blockNum.workMax;
    int kthValue = individualStruct->blockNum.kthValue;
    pthread_mutex_unlock(&mutex);

    for (size_t valueChecking = workMin; valueChecking <= workMax; valueChecking++) {
      if (globalPrimeArray[valueChecking] != 0) { // Then the value checking appears to be prime
        if (globalPrimeArray[valueChecking] == kthValue){
          // Do NOTHING
        } else if (globalPrimeArray[valueChecking]%kthValue == 0) {
          globalPrimeArray[valueChecking] = 0;
        }
      }
    } // END OF FOR LOOP

  } // END OF WHILE LOOP

  return 0;
}
