(define (problem tmp)
	(:domain workspace)
	(:objects
		rack - receptacle
		milk - box
	)
	(:init
		(aligned rack)
		(poslimit rack)
		(inworkspace rack)
		(on rack table)
		(on milk table)
	)
	(:goal (and))
)
