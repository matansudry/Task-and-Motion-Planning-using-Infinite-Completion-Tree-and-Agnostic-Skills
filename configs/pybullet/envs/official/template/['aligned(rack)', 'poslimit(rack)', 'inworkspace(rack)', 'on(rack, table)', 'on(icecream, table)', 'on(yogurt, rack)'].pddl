(define (problem tmp)
	(:domain workspace)
	(:objects
		rack - receptacle
		yogurt - box
		icecream - box
	)
	(:init
		(aligned rack)
		(poslimit rack)
		(inworkspace rack)
		(on rack table)
		(on icecream table)
		(on yogurt rack)
	)
	(:goal (and))
)
