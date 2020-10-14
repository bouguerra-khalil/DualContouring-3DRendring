

CIBLE = main
SRCS =  src/Camera.cpp main.cpp src/Trackball.cpp ann/src/ANN.cpp ann/src/brute.cpp ann/src/kd_tree.cpp ann/src/kd_util.cpp ann/src/kd_split.cpp \
	ann/src/kd_dump.cpp ann/src/kd_search.cpp ann/src/kd_pr_search.cpp ann/src/kd_fix_rad_search.cpp \
	ann/src/bd_tree.cpp ann/src/bd_search.cpp ann/src/bd_pr_search.cpp ann/src/bd_fix_rad_search.cpp \
	ann/src/perf.cpp
LIBS =  -L/usr/lib -lglut -lGLU -lGL -lm -lpthread -lgsl -lgslcblas

#########################################################"

INCDIR = .
LIBDIR = .
BINDIR = .

CC = g++
CPP = g++

CFLAGS = -Wall -O3
CXXFLAGS = -Wall -O3

CPPFLAGS =  -I$(INCDIR) -Iann/include

LDFLAGS = -L/usr/X11R6/lib
LDLIBS = -L$(LIBDIR) $(LIBS)


OBJS = $(SRCS:.cpp=.o)

$(CIBLE): $(OBJS)

install:  $(CIBLE)
	cp $(CIBLE) $(BINDIR)/

installdirs:
	test -d $(INCDIR) || mkdir $(INCDIR)
	test -d $(LIBDIR) || mkdir $(LIBDIR)
	test -d $(BINDIR) || mkdir $(BINDIR)

clean:
	rm -f  *~  $(CIBLE) $(OBJS)

veryclean: clean
	rm -f $(BINDIR)/$(CIBLE)

dep:
	gcc $(CPPFLAGS) -MM $(SRCS)

Camera.o: src/Camera.cpp src/Camera.h src/Vec3.h src/Trackball.h
main.o: main.cpp src/Vec3.h src/Camera.h src/Trackball.h
Trackball.o: src/Trackball.cpp src/Trackball.h
