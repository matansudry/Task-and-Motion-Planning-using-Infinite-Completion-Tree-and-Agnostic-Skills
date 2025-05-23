(define (problem tmp)
	(:domain workspace)
	(:objects
		rack - receptacle
		milk - box
		salt - box
	)
	(:init
		(aligned rack)
		(poslimit rack)
		(inworkspace rack)
		(on rack table)
		(on milk table)
		(inhand salt)
	)
	(:goal (and))
)
