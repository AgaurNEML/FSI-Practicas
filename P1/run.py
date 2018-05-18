# Search methods

import search

ab = search.GPSProblem('A', 'B', search.romania)
at = search.GPSProblem('A', 'T', search.romania)
ar = search.GPSProblem('A', 'R', search.romania)
aS = search.GPSProblem('A', 'S', search.romania)
ba = search.GPSProblem('B', 'A', search.romania)
gh = search.GPSProblem('G', 'H', search.romania)
cd = search.GPSProblem('C', 'D', search.romania)


print "\nBUSQUEDA POR RAMIFICACION Y SALTO\n"
print search.search_ramification(ab).path()
print search.search_heuristic(ab).path()
print "A -> B"

print search.search_ramification(at).path()
print "A -> T"

print search.search_ramification(ar).path()
print "A -> R"

print search.search_ramification(aS).path()
print "A -> S"

print search.search_ramification(ba).path()
print "B -> A"

print search.search_ramification(gh).path()
print "G -> H"

print search.search_ramification(cd).path()
print "C -> D"

print '\n---------------------------------------'
print '---------------------------------------'

print '\nBUSQUEDA INFORMADA(BUSQUEDA POR RAMIFICACION Y SALTO CON SUBESTIMACION)\n'
print search.search_heuristic(ab).path()
print "A -> B"

print search.search_heuristic(at).path()
print "A -> T"

print search.search_heuristic(ar).path()
print "A -> R"

print search.search_heuristic(aS).path()
print "A -> S"

print search.search_heuristic(ba).path()
print "B -> A"

print search.search_heuristic(gh).path()
print "G -> H"

print search.search_heuristic(cd).path()
print "C -> D"