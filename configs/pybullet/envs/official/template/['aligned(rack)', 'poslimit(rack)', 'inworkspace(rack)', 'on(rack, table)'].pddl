(define (problem tmp)
	(:domain workspace)
	(:objects
		rack - receptacle
	)
	(:init
		(aligned rack)
		(poslimit rack)
		(inworkspace rack)
		(on rack table)
	)
	(:goal (and))
)
