(define (problem tmp)
	(:domain workspace)
	(:objects
		rack - receptacle
		yogurt - box
		salt - box
	)
	(:init
		(aligned rack)
		(poslimit rack)
		(inworkspace rack)
		(on rack table)
		(on salt rack)
		(on yogurt table)
	)
	(:goal (and))
)
