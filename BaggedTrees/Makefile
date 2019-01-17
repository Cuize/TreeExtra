SHAREDDIR=../shared
LIBDIR=../ThreadPool
CXXFLAGS = -I$(SHAREDDIR) -I$(LIBDIR)
OBJS = Tree.o $(SHAREDDIR)/SplitInfo.o  $(SHAREDDIR)/INDdata.o $(SHAREDDIR)/TreeNode.o bt_functions.o $(SHAREDDIR)/functions.o $(SHAREDDIR)/LogStream.o $(LIBDIR)/thread_pool.o
PGMS = gbt_train
PGMOBJS = gbt_train.o
LIBS = -lpthread

all: directories $(PGMS)

directories:
	mkdir -p ../Bin

clean:
	rm $(OBJS)
	rm $(PGMOBJS)
	rm $(PGMS)

.cpp.o:
	g++ -O3 $(CXXFLAGS) -c $< -o $@

$(SHAREDDIR)/INDdata.o: $(SHAREDDIR)/definitions.h

gbt_train.o: $(OBJS)

gbt_train: gbt_train.o $(OBJS)
	g++ -O3 -o ../Bin/gbt_train gbt_train.o $(OBJS) $(LIBS)

