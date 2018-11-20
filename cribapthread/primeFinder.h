// BOTH FILES
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include <time.h>

//                                                       [pthreads-h3.c]
//--------------------------------------------------------------------->
struct work {
  int workMin;
  int workMax;
  int kthValue;
};
struct threadStruct {
  pthread_t id;
  int threadIndex;
  struct work blockNum;
};

//                                                           [CONSTANTS]
//--------------------------------------------------------------------->
#define FAIL 0
#define PASS 1
#define NOT_DONE 0
#define DONE 1
enum ASCII{ZERO_ASCII = 48, NINE_ASCII = 57};

//                                            PROTOTYPES [pthreads-h3.c]
//--------------------------------------------------------------------->
int validateCommandLine(int argc, char *argv[]);
int checkForOnlyNumbers(int argc, char * argv[]);
int secondParameter(char * argv[]);
int thirdParameter(char * argv[]);

void createWork(struct work **returnWorkPackets, int ARRAY_LENGTH, int NUMBER_OF_WORK_PACKETS, int chunkSize);
void getWork(struct work **returnWorkPacket);
void* workMethod(void* p);

// GLOBAL VARIABLES
pthread_mutex_t mutex;

// Needed
int *globalPrimeArray;
struct work* workSegment;
int UP_TO_THIS_NUMBER; // Used in workMethod()
int kthValueON; // Used in getWork()
int packetON; // Used in getWork()
int packetsLEFT; // Used in getWork()
int resetPACKETS; // Used in getWork()
