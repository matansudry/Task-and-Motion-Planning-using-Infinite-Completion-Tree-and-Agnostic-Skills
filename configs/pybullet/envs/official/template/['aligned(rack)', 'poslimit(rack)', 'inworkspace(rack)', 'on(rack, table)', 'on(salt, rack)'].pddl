(define (problem tmp)
	(:domain workspace)
	(:objects
		rack - receptacle
		salt - box
	)
	(:init
		(aligned rack)
		(poslimit rack)
		(inworkspace rack)
		(on rack table)
		(on salt rack)
	)
	(:goal (and))
)
