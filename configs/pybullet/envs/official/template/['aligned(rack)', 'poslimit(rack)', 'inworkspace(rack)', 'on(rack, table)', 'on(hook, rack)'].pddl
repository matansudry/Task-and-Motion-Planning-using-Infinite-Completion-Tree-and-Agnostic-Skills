(define (problem tmp)
	(:domain workspace)
	(:objects
		rack - receptacle
		hook - tool
	)
	(:init
		(aligned rack)
		(poslimit rack)
		(inworkspace rack)
		(on rack table)
		(on hook rack)
	)
	(:goal (and))
)
