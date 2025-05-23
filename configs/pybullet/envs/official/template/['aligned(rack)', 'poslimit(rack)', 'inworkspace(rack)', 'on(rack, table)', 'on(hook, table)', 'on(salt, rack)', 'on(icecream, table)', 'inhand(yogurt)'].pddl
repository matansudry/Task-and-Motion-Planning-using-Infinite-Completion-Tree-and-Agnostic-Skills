(define (problem tmp)
	(:domain workspace)
	(:objects
		rack - receptacle
		hook - tool
		yogurt - box
		icecream - box
		salt - box
	)
	(:init
		(aligned rack)
		(poslimit rack)
		(inworkspace rack)
		(on rack table)
		(on hook table)
		(on salt rack)
		(on icecream table)
		(inhand yogurt)
	)
	(:goal (and))
)
