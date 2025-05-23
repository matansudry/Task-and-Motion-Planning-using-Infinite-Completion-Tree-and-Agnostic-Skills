(define (problem tmp)
	(:domain workspace)
	(:objects
		rack - receptacle
		hook - tool
		salt - box
	)
	(:init
		(aligned rack)
		(poslimit rack)
		(inworkspace rack)
		(on rack table)
		(on salt table)
		(inhand hook)
	)
	(:goal (and))
)
